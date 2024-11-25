import re
import copy
import ast
import enum
import inspect
import sys
import numpy as np

from google.protobuf import message
from google.protobuf import text_format


def ToFlattenedTextDict(p, quote_all=True):
  """Flattens entries in 'p' (dict, Params) in textual format."""
  kv = {}
  types = {}

  def GetRepr(val):
    """Get the representation of `val`."""
    if isinstance(val, Params):
      return _SortedDict({k: GetRepr(v) for k, v in val.IterParams()})
    if isinstance(val, dict):
      return _SortedDict({k: GetRepr(v) for k, v in val.items()})
    # if dataclasses.is_dataclass(val):
    #   return _SortedDict({k: GetRepr(v) for k, v in val.__dict__.items()})
    if _IsNamedTuple(val):
      return _SortedDict({k: GetRepr(v) for k, v in val._asdict().items()})
    if isinstance(val, (list, tuple)):
      return type(val)([GetRepr(v) for v in val])
    if isinstance(val, (int, float, bool, str, enum.Enum)):
      return val
    if np.isscalar(val):
      return float(val.item())
    # if isinstance(val, tf.DType):
    #   return val.name
    if isinstance(val, message.Message):
      proto_str = text_format.MessageToString(val, as_one_line=True)
      return 'proto/%s/%s/%s' % (inspect.getmodule(val).__name__,
                                  type(val).__name__, proto_str)
    if isinstance(val, type):
      return 'type/' + inspect.getmodule(val).__name__ + '/' + val.__name__
    return type(val).__name__

  def Traverse(p, prefix, kv):
    """Traverses 'p' and inserts key-value pairs to 'kv'."""
    if isinstance(p, dict):
      for key, val in p.items():
        Traverse(val, prefix + '.' + key if len(prefix) else key, kv)
    elif isinstance(p, Params):
      for key, val in p.IterParams():
        Traverse(val, prefix + '.' + key if len(prefix) else key, kv)
    elif (isinstance(p, (list, tuple)) and
          all(isinstance(x, Params) for x in p)):
      for i, val in enumerate(p):
        Traverse(val, '%s[%d]' % (prefix, i), kv)
    elif isinstance(p, str):
      kv[prefix] = _QuoteString(p) if quote_all else p
      types[prefix[1:]] = 'str'
    else:
      kv[prefix] = str(GetRepr(p)) if quote_all else GetRepr(p)
      types[prefix[1:]] = type(p).__name__

  Traverse(p, '', kv)
  return kv, types


def _QuoteString(s):
  """Quotes a string with appropriate quotes and escaping.

  This performs lite escaping by choosing enclosing quotation marks that would
  escape the least (either single or double quotes) and escaping those quotes
  and the backslash. Note that this does not escape newlines. If the string
  contains embedded newlines, they will be output verbatim.

  Args:
    s: String to quote.
  Returns:
    Quotes string (possibly multiline).
  """
  single_quote_count = s.count('\'')
  double_quote_count = s.count('"')
  quote_delim = '\'' if single_quote_count <= double_quote_count else '"'
  # Apply escaping to the chosen quote character and the backslash.
  encoded = re.sub(r'([%s\\])' % quote_delim, r'\\\1', s)
  return quote_delim + encoded + quote_delim


def _UnquoteString(quoted):
  if quoted and quoted[0] in ['"', '\'']:
    # Note that only the limited set of escaping produced by _QuoteString is
    # supported.
    contents = quoted.strip(quoted[0])
    return re.sub(r"""\\([\\'"])""", r'\1', contents)
  else:
    # Just return literal text.
    return quoted


def _EndsWithTerminalQuote(s, quote_char):
  """Returns whether a string ends with a valid terminal quote."""
  endm = re.search(r'(\\*)%s$' % quote_char, s)
  if not endm:
    return False
  backslashes = endm.group(1)
  if len(backslashes) % 2 == 0:
    # Even number of backslashes preceding the quote means the quote is
    # not escaped.
    return True
  else:
    # Terminal quote is escaped.
    return False



class _SortedDict(dict):
  """A dict with a __repr__ that is always sorted by key."""

  def __repr__(self):
    return '{' + ', '.join(
        '%r: %r' % item for item in sorted(self.items())) + '}'
  

def _IsNamedTuple(x):
  """Returns whether an object is an instance of a collections.namedtuple.

  Examples::

    _IsNamedTuple((42, 'hi')) ==> False
    Foo = collections.namedtuple('Foo', ['a', 'b'])
    _IsNamedTuple(Foo(a=42, b='hi')) ==> True

  Args:
    x: The object to check.
  """
  return isinstance(x, tuple) and hasattr(x, '_fields')
  

class _Param:
  """Stores data for a single parameter."""

  def __init__(self, name, default_value, description):
    self._name = name
    self._value = default_value
    self._description = description
    
  def __eq__(self, other):
    # pylint: disable=protected-access
    return self._name == other._name and self._value == other._value
  
  def __deepcopy__(self, memo):
    if False: # isinstance(self._value, (tf.Tensor, symbolic.Symbol)):
      # In case self._value is a tensor/symbol, let's just make a reference.
      value = self._value
    else:
      value = copy.deepcopy(self._value, memo)
    p = _Param(self._name, value, self._description)
    # Q(yonghui): Is this the right use of memo.
    memo[id(self)] = p
    return p
  
  def ToString(self, nested_depth):
    """Prints the parameter as a string."""

    def GetRepr(val):
      """Get the representation of `val`."""
      if isinstance(val, Params):
        return _SortedDict({k: GetRepr(v) for k, v in val.IterParams()})
      if isinstance(val, dict):
        return _SortedDict({k: GetRepr(v) for k, v in val.items()})
      if isinstance(val, (list, tuple)) and not _IsNamedTuple(val):
        # NB: this constructor signature works for tuples, but not namedtuples.
        return type(val)([GetRepr(v) for v in val])
      # NOTE(markmurphy): I introduced Repr() because it's impossible (afaik) to
      # overwrite the __str__ or __repr__ method of a types.FunctionType object.
      if hasattr(val, 'Repr'):
        return val.Repr()
      return val

    nested_indent = '  ' * nested_depth
    if isinstance(self._value, Params):
      # pylint: disable=protected-access
      value_str = self._value._ToString(nested_depth)
    elif isinstance(self._value, str):
      return '%s%s: "%s"' % (nested_indent, self._name, self._value)
    else:
      value_str = str(GetRepr(self._value))
    return '%s%s: %s' % (nested_indent, self._name, value_str)
  
  def Set(self, value):
    # Note that we don't make a copy of Params objects.
    # TODO(sadovsky): Maybe add safeguard to ensure that Params object is not
    # owned by other Params objects.
    self._value = value

  def Get(self):
    return self._value
  

def CopyFieldsTo(from_p, to_p, skip=None):
  """Copy fields from one Params to another, with optional skipped params.

  Preserves `type(to_p.Instantiate())`. Use `from_p.Copy()` instead if requiring
  a deep copy of `from_p`, without updating `to_p`.

  Args:
    from_p: Source params to copy from.
    to_p: Destination params to copy to.
    skip: If not None, a list of strings of param names to skip. Automatically
      skips InstantiableParams' 'cls' parameter.

  Returns:
    The updated to_p.
  """
  skip = skip or []
  skip.append('cls')
  for n, p in from_p.IterParams():
    if n in skip:
      continue
    if isinstance(p, Params):
      to_p.Set(**{n: p.Copy()})
    else:
      to_p.Set(**{n: p})
  return to_p
  

class Params:
  """Stores data for a set of parameters.

  Provides attribute-based API, e.g. "params.foo = 5".
  Uses internal {'name': _Param} dict for storing parameter data.
  """

  def __init__(self):
    self.__dict__['_immutable'] = False
    self._params = {}  # name => _Param
    
  def __setattr__(self, name, value):
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    if name == '_params' or name == '_immutable':
      self.__dict__[name] = value
    else:
      try:
        self._params[name].Set(value)
      except KeyError:
        raise AttributeError(self._KeyErrorString(name))

  def __getattr__(self, name):
    if name == '_params' or name == '_immutable':
      return self.__dict__[name]
    try:
      return self._params[name].Get()
    except KeyError:
      # cPickle expects __getattr__ to raise AttributeError, not KeyError.
      raise AttributeError(self._KeyErrorString(name))

  def __dir__(self):
    return sorted(self._params.keys())

  def __contains__(self, name):
    return name in self._params

  def __len__(self):
    return len(self._params)
  
  def __eq__(self, other):
    return isinstance(other, Params) and self._params == other._params  # pylint: disable=protected-access

  def __ne__(self, other):
    return not self == other

  def __str__(self):
    return self._ToString(0)

  def _ToString(self, nested_depth):
    # Note: We use iteritems() below so as to sort by name.
    sorted_param_strs = [
        v.ToString(nested_depth + 1) for (_, v) in sorted(self._params.items())
    ]
    nested_indent = '  ' * nested_depth
    return '{\n%s\n%s}' % ('\n'.join(sorted_param_strs), nested_indent)
  
  def __deepcopy__(self, unused_memo):
    return self.Copy()
  
  def _SimilarKeys(self, name):
    """Return a list of params keys that are similar to name."""

    def _Overlaps(name, key):
      """The fraction of 3-char substrings in <name> that appear in key."""
      matches = 0
      trials = 0
      for i in range(len(name) - 3):
        trials += 1
        if name[i:i + 3] in key:
          matches += 1
      if trials:
        return float(matches) / trials
      return 0

    if '_params' in self.__dict__:
      return [key for key in self._params if _Overlaps(name, key) > 0.5]
    return []

  def _KeyErrorString(self, name):
    similar = self._SimilarKeys(name)
    if similar:
      return name + ' (did you mean: [%s])' % (','.join(sorted(similar)))
    return name
  
  def Copy(self):
    """Creates a deep copy of self."""
    return self._CopyTo(type(self)())

  def _CopyTo(self, res):
    # pylint: disable=protected-access
    res._params = copy.deepcopy(self._params)
    res._immutable = self._immutable
    # pylint: enable=protected-access
    return res
  
  def Define(self, name, default_value, description):
    """Defines a parameter.

    Args:
      name: The parameter name. Must only contain lowercase letters, numbers,
          and underscores. Must start with lowercase letter.
      default_value: Default value for this parameter. May be None.
      description: String description of this parameter.

    Raises:
      AttributeError: If parameter 'name' is already defined.
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    assert name is not None and isinstance(
        name, str) and (re.match('^[a-z][a-z0-9_]*$', name) is not None)
    if name in self._params:
      raise AttributeError('Parameter %s is already defined' % name)
    self._params[name] = _Param(name, default_value, description)
    
  def Freeze(self):
    """Marks this Params as immutable."""
    self._immutable = True

  def IsImmutable(self):
    """Return whether this Params is immutable."""
    return self._immutable

  def _GetNested(self, name):
    """Returns nested param by its name."""
    parts = name.split('.')
    curr = self
    for i, part in enumerate(parts[:-1]):
      # Get the value (nested Params object) associated with name 'part'.
      try:
        is_list = re.match(r'^(.+)\[(.+)\]$', part)
        if is_list:
          part = is_list.group(1)
          list_index = int(is_list.group(2))
        # pylint: disable=protected-access
        curr = curr._params[part].Get()
        if is_list:
          curr = curr[list_index]
      except KeyError:
        raise AttributeError('.'.join(parts[:i + 1]))
      assert isinstance(curr, Params), (
          'Cannot introspect %s for %s' % (type(curr), '.'.join(parts[:i + 1])))
    return curr, parts[-1]
    
  def Set(self, **kwargs):
    """Sets multiple parameters.

    Dots in names indicate navigation into nested Params objects. We do not
    allow navigation into lists or dicts, and may ban these types altogether in
    favor of string representations.

    Args:
      **kwargs: Name-value pairs to set.

    Returns:
      self
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable: %s' % self)
    for name, value in kwargs.items():
      # Get nested param.
      param, key = self._GetNested(name)
      # Update the value associated with key.
      try:
        # pylint: disable=protected-access
        param._params[key].Set(value)
      except KeyError:
        raise AttributeError(self._KeyErrorString(name))
    return self
  
  def Get(self, name):
    """Get parameter.

    Dots in names indicate navigation into nested Params objects. We do not
    allow navigation into lists or dicts, and may ban these types altogether in
    favor of string representations.

    Args:
      name: (str) Name.

    Returns:
      value.

    Raises:
      AttributeError: if parameter is not found
    """
    param, key = self._GetNested(name)
    # Get the value associated with key.
    try:
      # pylint: disable=protected-access
      return param._params[key].Get()
    except KeyError:
      raise AttributeError(self._KeyErrorString(name))

  def Delete(self, *args):
    """Deletes multiple parameters.

    Dots in names indicate navigation into nested Params objects. We do not
    allow navigation into lists or dicts, and may ban these types altogether in
    favor of string representations.

    Args:
      *args: List of names.

    Returns:
      self
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    for name in args:
      # Get nested param.
      param, key = self._GetNested(name)
      # Delete the key.
      try:
        # pylint: disable=protected-access
        del param._params[key]
      except KeyError:
        raise AttributeError(self._KeyErrorString(name))
    return self
  
  def IterParams(self):
    """Pythonic dict-like iteration."""
    for name, param in self._params.items():
      yield (name, param.Get())

  def __iter__(self):
    """Allows treating this class as a Python dict."""
    return self.IterParams()
  
  def ToText(self, include_types=False):
    """Encodes params into a simple text format.

    Each param is represented as a single line in the output.  The param
    name and value is separated by a ":".  The nest param name is
    separated by ".".  For values of non-trivial types (types other than
    int, float, bool, str, and a few, etc.), we just print out the name
    of its type.

    Note that strings are enclosed in appropriate single or double quotes
    (whichever would involve the least escaping) and will have some characters
    backslash escaped. String properties can span multiple lines.

    Args:
      include_types: Should we return types of the values. If True, the types
        dict will be returned as a second val in a return tuple

    Returns:
      The encoded text or (encoded text, types dict) if include_types is True.
    """
    kv, types = ToFlattenedTextDict(self)
    ret = ''
    for (k, v) in sorted(kv.items()):
      # ret += k[1:] + ' : ' + v + '\n'
      ret += k + ' : ' + v + '\n'

    return (ret, types) if include_types else ret

  def FromText(self, text, type_overrides=None):
    """Merges params specified in 'text' into 'params'.

    'text' follows the simple text format as produced by
    ParamsToSimpleText.  For a param specified in both 'params' and
    'text', overwrites the value in 'params' according to 'text'.
    Params specified in 'text' but not in 'params' are ignored.

    Args:
      text: A text representation of params.
      type_overrides: Overrides for the types of the params.
    Raises:
      AttributeError: text contains invalid parameter key
      ValueError: text contains invalid parameter value, or the format is
                  wrong.
    """
    if self._immutable:
      raise TypeError('This Params instance is immutable.')
    kv = {}
    type_overrides = type_overrides or {}
    string_continue = None  # None or (key, quote, value)
    for line in text.split('\n'):
      # Continuing a multi-line string.
      if string_continue:
        value_stripped = line.rstrip()
        if not _EndsWithTerminalQuote(value_stripped, string_continue[1]):
          # String continues
          string_continue = (string_continue[0], string_continue[1],
                             string_continue[2] + '\n' + line)
          continue
        # String terminates.
        kv[string_continue[0]] = string_continue[2] + '\n' + value_stripped
        string_continue = None
        continue

      # Regular line.
      line = line.strip()
      if not line or line[0] == '#':
        # empty line or comment
        continue
      pair = line.split(':', 1)
      if len(pair) == 2:
        key = pair[0].strip()
        value = pair[1].lstrip()
        value_stripped = value.rstrip()
        # Detect single vs multi-line string start.
        if value and value[0] in ['"', '\'']:
          quote_char = value[0]
          if not _EndsWithTerminalQuote(value[1:], quote_char):
            # Multi-line string.
            string_continue = (key, quote_char, value)
            continue
        kv[key] = value_stripped
      else:
        raise ValueError('Line {} is not in <key>:<value> format'.format(line))

    def _ValueFromText(key, old_val, val):
      """Returns the new param value from its text representation."""
      val_type = type(old_val).__name__
      if isinstance(old_val, str):
        val_type = 'str'
      if key in type_overrides:
        val_type = type_overrides[key]
      # Converts val (a string) to a best-guessed typed value.
      if val_type == 'bool':
        return val and (val != 'False') and (val != 'false')
      elif val_type == 'int':
        return int(val)
      elif val_type == 'float':
        return float(val)
      elif val_type in ['list', 'tuple']:
        return ast.literal_eval(val)
      elif val_type == 'dict':
        return ast.literal_eval(val) if val != 'dict' else {}
      elif val_type == 'str':
        val = _UnquoteString(val)
        if val.startswith('[') and val.endswith(']'):
          # We may have stored a list as a string, try converting to a list.
          # In case of ValueError - use the string as is.
          try:
            return ast.literal_eval(val)
          except ValueError:
            pass
        return val
      elif isinstance(old_val, enum.Enum):
        cls, _, name = val.rpartition('.')
        if val_type != cls:
          raise ValueError('Expected enum of class %s but got %s' %
                           (val_type, cls))
        return type(old_val)[name]
      elif (isinstance(old_val, type) or isinstance(old_val, message.Message) or
            old_val is None):
        if val == 'NoneType':
          return None
        elif old_val is None and val in ('False', 'false'):
          return False
        elif old_val is None and val in ('True', 'true'):
          return True
        else:
          try:
            val_type, pkg, cls = val.split('/', 2)
            if val_type == 'type':
              return getattr(sys.modules[pkg], cls)
            elif val_type == 'proto':
              cls, proto_str = cls.split('/', 1)
              proto_cls = getattr(sys.modules[pkg], cls)
              if not issubclass(proto_cls, message.Message):
                raise ValueError('%s is not a proto class.' % proto_cls)
              return text_format.Parse(proto_str, proto_cls())
          except ValueError as e:
            raise ValueError('Error processing %r : %r with %r' % (key, val, e))
      else:
        raise ValueError('Failed to read a parameter: %r : %r' % (key, val))

    for key, val in kv.items():
      old_val = self.Get(key)
      new_val = _ValueFromText(key, old_val, val)
      self.Set(**{key: new_val})

  def ToTextWithTypes(self):
    """Same as ToText but encodes both params and their types."""
    text, types = self.ToText(include_types=True)
    text += '\n\n'
    for (k, v) in sorted(types.items()):
      text += k + ' : ' + v + '\n'
    return text

  def FromTextWithTypes(self, text):
    """Same as FromText but expects to have types encoded in the text."""
    text, types_str = text.split('\n\n')
    types = {}
    for row in types_str.split('\n'):
      if not row:
        continue
      k, v = row.split(':')
      types[k.strip()] = v.strip()
    self.FromText(text, type_overrides=types)

  def TextDiff(self, other):
    """Return the differences between this object and another as a string.

    Args:
      other: The other Params object.

    Returns:
      A string of differences.
    """

    def TextDiffHelper(a, b, spaces):
      """Return the differences between a and b as a string."""
      a_keys = set([key for key, _ in a.IterParams()])
      b_keys = set([key for key, _ in b.IterParams()])
      all_keys = a_keys.union(b_keys)
      diff = ''
      for key in sorted(all_keys):
        if key in a_keys and key not in b_keys:
          diff += '>' + spaces + key + ': ' + str(a.Get(key)) + '\n'
        elif key in b_keys and key not in a_keys:
          diff += '<' + spaces + key + ': ' + str(b.Get(key)) + '\n'
        elif a.Get(key) != b.Get(key):
          if isinstance(a.Get(key), Params):
            diff += '?' + spaces + key + ':\n'
            diff += TextDiffHelper(a.Get(key), b.Get(key), spaces + '  ')
          else:
            diff += '>' + spaces + key + ': ' + str(a.Get(key)) + '\n'
            diff += '<' + spaces + key + ': ' + str(b.Get(key)) + '\n'
      return diff

    return TextDiffHelper(self, other, spaces=' ')
    

class InstantiableParams(Params):
  """Params which can be instantiated.

  When using InstantiableParams, callers must provide a class which supports
  initialization using a Params instance.

  This covers a common use case of Params to hold a configuration for a given
  class.
  """

  def __init__(self, cls=None):
    super().__init__()
    self.Define('cls', cls, 'Cls that this param object is associated with.')
    
  def Copy(self):
    """See base class."""
    return self._CopyTo(type(self)(self.cls))


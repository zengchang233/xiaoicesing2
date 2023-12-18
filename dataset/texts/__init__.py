""" from https://github.com/keithito/tacotron """
import re
# import sys
# sys.path.insert(0, '/home/zengchang/code/acoustic_v2')
from dataset.texts import cleaners
from dataset.texts.symbols import symbols

from ipdb import set_trace

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through

    Returns:
      List of integers corresponding to the symbols in the text
    """
    # set_trace()
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    # while len(text):
        # m = _curly_re.match(text)

        # if not m:
        #     clean_text = _clean_text(text, cleaner_names).split(' ')
        #     sequence += _symbols_to_sequence(clean_text)
        #     break
        # clean_text1 = _clean_text(m.group(1), cleaner_names).split(' ')
        # sequence += _symbols_to_sequence(clean_text1)
        # clean_text2 = m.group(2).split(' ')
        # sequence += _arpabet_to_sequence(clean_text2)
        # text = m.group(3)
    sequence += _arpabet_to_sequence(text)

    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")

def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text

def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]

def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])

def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"

if __name__ == "__main__":
    text = 'Turn left on {HH AW1 S S T AH0 N} Street.'
    print(text_to_sequence(text, ['english_cleaners']))
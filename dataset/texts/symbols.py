""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from dataset.texts import cmudict, pinyin

_pad = "_~"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]
# _silences = ["sp", "spn", "sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = [s for s in cmudict.valid_symbols]
_arpabet = ["@" + s for s in cmudict.valid_symbols]
# _pinyin = [s for s in pinyin.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)

# symbols
'''
['_',
 '~',
 '-',
 '!',
 "'",
 '(',
 ')',
 ',',
 '.',
 ':',
 ';',
 '?',
 ' ',
 'A',
 'B',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'J',
 'K',
 'L',
 'M',
 'N',
 'O',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'U',
 'V',
 'W',
 'X',
 'Y',
 'Z',
 'a',
 'b',
 'c',
 'd',
 'e',
 'f',
 'g',
 'h',
 'i',
 'j',
 'k',
 'l',
 'm',
 'n',
 'o',
 'p',
 'q',
 'r',
 's',
 't',
 'u',
 'v',
 'w',
 'x',
 'y',
 'z',
 '@AA',
 '@AA0',
 '@AA1',
 '@AA2',
 '@AE',
 '@AE0',
 '@AE1',
 '@AE2',
 '@AH',
 '@AH0',
 '@AH1',
 '@AH2',
 '@AO',
 '@AO0',
 '@AO1',
 '@AO2',
 '@AW',
 '@AW0',
 '@AW1',
 '@AW2',
 '@AY',
 '@AY0',
 '@AY1',
 '@AY2',
 '@B',
 '@CH',
 '@D',
 '@DH',
 '@EH',
 '@EH0',
 '@EH1',
 '@EH2',
 '@ER',
 '@ER0',
 '@ER1',
 '@ER2',
 '@EY',
 '@EY0',
 '@EY1',
 '@EY2',
 '@F',
 '@G',
 '@HH',
 '@IH',
 '@IH0',
 '@IH1',
 '@IH2',
 '@IY',
 '@IY0',
 '@IY1',
 '@IY2',
 '@JH',
 '@K',
 '@L',
 '@M',
 '@N',
 '@NG',
 '@OW',
 '@OW0',
 '@OW1',
 '@OW2',
 '@OY',
 '@OY0',
 '@OY1',
 '@OY2',
 '@P',
 '@R',
 '@S',
 '@SH',
 '@T',
 '@TH',
 '@UH',
 '@UH0',
 '@UH1',
 '@UH2',
 '@UW',
 '@UW0',
 '@UW1',
 '@UW2',
 '@V',
 '@W',
 '@Y',
 '@Z',
 '@ZH',
 '@b',
 '@c',
 '@ch',
 '@d',
 '@f',
 '@g',
 '@h',
 '@j',
 '@k',
 '@l',
 '@m',
 '@n',
 '@p',
 '@q',
 '@r',
 '@s',
 '@sh',
 '@t',
 '@w',
 '@x',
 '@y',
 '@z',
 '@zh',
 '@a1',
 '@a2',
 '@a3',
 '@a4',
 '@a5',
 '@ai1',
 '@ai2',
 '@ai3',
 '@ai4',
 '@ai5',
 '@an1',
 '@an2',
 '@an3',
 '@an4',
 '@an5',
 '@ang1',
 '@ang2',
 '@ang3',
 '@ang4',
 '@ang5',
 '@ao1',
 '@ao2',
 '@ao3',
 '@ao4',
 '@ao5',
 '@e1',
 '@e2',
 '@e3',
 '@e4',
 '@e5',
 '@ei1',
 '@ei2',
 '@ei3',
 '@ei4',
 '@ei5',
 '@en1',
 '@en2',
 '@en3',
 '@en4',
 '@en5',
 '@eng1',
 '@eng2',
 '@eng3',
 '@eng4',
 '@eng5',
 '@er1',
 '@er2',
 '@er3',
 '@er4',
 '@er5',
 '@i1',
 '@i2',
 '@i3',
 '@i4',
 '@i5',
 '@ia1',
 '@ia2',
 '@ia3',
 '@ia4',
 '@ia5',
 '@ian1',
 '@ian2',
 '@ian3',
 '@ian4',
 '@ian5',
 '@iang1',
 '@iang2',
 '@iang3',
 '@iang4',
 '@iang5',
 '@iao1',
 '@iao2',
 '@iao3',
 '@iao4',
 '@iao5',
 '@ie1',
 '@ie2',
 '@ie3',
 '@ie4',
 '@ie5',
 '@ii1',
 '@ii2',
 '@ii3',
 '@ii4',
 '@ii5',
 '@iii1',
 '@iii2',
 '@iii3',
 '@iii4',
 '@iii5',
 '@in1',
 '@in2',
 '@in3',
 '@in4',
 '@in5',
 '@ing1',
 '@ing2',
 '@ing3',
 '@ing4',
 '@ing5',
 '@iong1',
 '@iong2',
 '@iong3',
 '@iong4',
 '@iong5',
 '@iou1',
 '@iou2',
 '@iou3',
 '@iou4',
 '@iou5',
 '@o1',
 '@o2',
 '@o3',
 '@o4',
 '@o5',
 '@ong1',
 '@ong2',
 '@ong3',
 '@ong4',
 '@ong5',
 '@ou1',
 '@ou2',
 '@ou3',
 '@ou4',
 '@ou5',
 '@u1',
 '@u2',
 '@u3',
 '@u4',
 '@u5',
 '@ua1',
 '@ua2',
 '@ua3',
 '@ua4',
 '@ua5',
 '@uai1',
 '@uai2',
 '@uai3',
 '@uai4',
 '@uai5',
 '@uan1',
 '@uan2',
 '@uan3',
 '@uan4',
 '@uan5',
 '@uang1',
 '@uang2',
 '@uang3',
 '@uang4',
 '@uang5',
 '@uei1',
 '@uei2',
 '@uei3',
 '@uei4',
 '@uei5',
 '@uen1',
 '@uen2',
 '@uen3',
 '@uen4',
 '@uen5',
 '@uo1',
 '@uo2',
 '@uo3',
 '@uo4',
 '@uo5',
 '@v1',
 '@v2',
 '@v3',
 '@v4',
 '@v5',
 '@van1',
 '@van2',
 '@van3',
 '@van4',
 '@van5',
 '@ve1',
 '@ve2',
 '@ve3',
 '@ve4',
 '@ve5',
 '@vn1',
 '@vn2',
 '@vn3',
 '@vn4',
 '@vn5',
 '@rr',
 '@sp',
 '@spn',
 '@sil']
'''
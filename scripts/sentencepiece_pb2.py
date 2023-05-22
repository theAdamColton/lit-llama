# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: sp.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x08sp.proto\x12\rsentencepiece\"\x80\x0c\n\x0bTrainerSpec\x12\r\n\x05input\x18\x01 \x03(\t\x12\x14\n\x0cinput_format\x18\x07 \x01(\t\x12\x14\n\x0cmodel_prefix\x18\x02 \x01(\t\x12\x41\n\nmodel_type\x18\x03 \x01(\x0e\x32$.sentencepiece.TrainerSpec.ModelType:\x07UNIGRAM\x12\x18\n\nvocab_size\x18\x04 \x01(\x05:\x04\x38\x30\x30\x30\x12\x17\n\x0f\x61\x63\x63\x65pt_language\x18\x05 \x03(\t\x12 \n\x15self_test_sample_size\x18\x06 \x01(\x05:\x01\x30\x12*\n\x1b\x65nable_differential_privacy\x18\x32 \x01(\x08:\x05\x66\x61lse\x12+\n differential_privacy_noise_level\x18\x33 \x01(\x02:\x01\x30\x12\x32\n\'differential_privacy_clipping_threshold\x18\x34 \x01(\x04:\x01\x30\x12\"\n\x12\x63haracter_coverage\x18\n \x01(\x02:\x06\x30.9995\x12\x1e\n\x13input_sentence_size\x18\x0b \x01(\x04:\x01\x30\x12$\n\x16shuffle_input_sentence\x18\x13 \x01(\x08:\x04true\x12 \n\x14mining_sentence_size\x18\x0c \x01(\x05\x42\x02\x18\x01\x12\"\n\x16training_sentence_size\x18\r \x01(\x05\x42\x02\x18\x01\x12(\n\x17seed_sentencepiece_size\x18\x0e \x01(\x05:\x07\x31\x30\x30\x30\x30\x30\x30\x12\x1e\n\x10shrinking_factor\x18\x0f \x01(\x02:\x04\x30.75\x12!\n\x13max_sentence_length\x18\x12 \x01(\x05:\x04\x34\x31\x39\x32\x12\x17\n\x0bnum_threads\x18\x10 \x01(\x05:\x02\x31\x36\x12\x1d\n\x12num_sub_iterations\x18\x11 \x01(\x05:\x01\x32\x12$\n\x18max_sentencepiece_length\x18\x14 \x01(\x05:\x02\x31\x36\x12%\n\x17split_by_unicode_script\x18\x15 \x01(\x08:\x04true\x12\x1d\n\x0fsplit_by_number\x18\x17 \x01(\x08:\x04true\x12!\n\x13split_by_whitespace\x18\x16 \x01(\x08:\x04true\x12)\n\x1atreat_whitespace_as_suffix\x18\x18 \x01(\x08:\x05\x66\x61lse\x12+\n\x1c\x61llow_whitespace_only_pieces\x18\x1a \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x0csplit_digits\x18\x19 \x01(\x08:\x05\x66\x61lse\x12#\n\x19pretokenization_delimiter\x18\x35 \x01(\t:\x00\x12\x17\n\x0f\x63ontrol_symbols\x18\x1e \x03(\t\x12\x1c\n\x14user_defined_symbols\x18\x1f \x03(\t\x12\x16\n\x0erequired_chars\x18$ \x01(\t\x12\x1c\n\rbyte_fallback\x18# \x01(\x08:\x05\x66\x61lse\x12+\n\x1dvocabulary_output_piece_score\x18  \x01(\x08:\x04true\x12\x1e\n\x10hard_vocab_limit\x18! \x01(\x08:\x04true\x12\x1c\n\ruse_all_vocab\x18\" \x01(\x08:\x05\x66\x61lse\x12\x11\n\x06unk_id\x18( \x01(\x05:\x01\x30\x12\x11\n\x06\x62os_id\x18) \x01(\x05:\x01\x31\x12\x11\n\x06\x65os_id\x18* \x01(\x05:\x01\x32\x12\x12\n\x06pad_id\x18+ \x01(\x05:\x02-1\x12\x18\n\tunk_piece\x18- \x01(\t:\x05<unk>\x12\x16\n\tbos_piece\x18. \x01(\t:\x03<s>\x12\x17\n\teos_piece\x18/ \x01(\t:\x04</s>\x12\x18\n\tpad_piece\x18\x30 \x01(\t:\x05<pad>\x12\x1a\n\x0bunk_surface\x18, \x01(\t:\x05 \xe2\x81\x87 \x12+\n\x1ctrain_extremely_large_corpus\x18\x31 \x01(\x08:\x05\x66\x61lse\"5\n\tModelType\x12\x0b\n\x07UNIGRAM\x10\x01\x12\x07\n\x03\x42PE\x10\x02\x12\x08\n\x04WORD\x10\x03\x12\x08\n\x04\x43HAR\x10\x04*\t\x08\xc8\x01\x10\x80\x80\x80\x80\x02\"\xd1\x01\n\x0eNormalizerSpec\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x1c\n\x14precompiled_charsmap\x18\x02 \x01(\x0c\x12\x1e\n\x10\x61\x64\x64_dummy_prefix\x18\x03 \x01(\x08:\x04true\x12&\n\x18remove_extra_whitespaces\x18\x04 \x01(\x08:\x04true\x12 \n\x12\x65scape_whitespaces\x18\x05 \x01(\x08:\x04true\x12\x1e\n\x16normalization_rule_tsv\x18\x06 \x01(\t*\t\x08\xc8\x01\x10\x80\x80\x80\x80\x02\"y\n\x0cSelfTestData\x12\x33\n\x07samples\x18\x01 \x03(\x0b\x32\".sentencepiece.SelfTestData.Sample\x1a)\n\x06Sample\x12\r\n\x05input\x18\x01 \x01(\t\x12\x10\n\x08\x65xpected\x18\x02 \x01(\t*\t\x08\xc8\x01\x10\x80\x80\x80\x80\x02\"\xfe\x03\n\nModelProto\x12\x37\n\x06pieces\x18\x01 \x03(\x0b\x32\'.sentencepiece.ModelProto.SentencePiece\x12\x30\n\x0ctrainer_spec\x18\x02 \x01(\x0b\x32\x1a.sentencepiece.TrainerSpec\x12\x36\n\x0fnormalizer_spec\x18\x03 \x01(\x0b\x32\x1d.sentencepiece.NormalizerSpec\x12\x33\n\x0eself_test_data\x18\x04 \x01(\x0b\x32\x1b.sentencepiece.SelfTestData\x12\x38\n\x11\x64\x65normalizer_spec\x18\x05 \x01(\x0b\x32\x1d.sentencepiece.NormalizerSpec\x1a\xd2\x01\n\rSentencePiece\x12\r\n\x05piece\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\x12\x42\n\x04type\x18\x03 \x01(\x0e\x32,.sentencepiece.ModelProto.SentencePiece.Type:\x06NORMAL\"T\n\x04Type\x12\n\n\x06NORMAL\x10\x01\x12\x0b\n\x07UNKNOWN\x10\x02\x12\x0b\n\x07\x43ONTROL\x10\x03\x12\x10\n\x0cUSER_DEFINED\x10\x04\x12\x08\n\x04\x42YTE\x10\x06\x12\n\n\x06UNUSED\x10\x05*\t\x08\xc8\x01\x10\x80\x80\x80\x80\x02*\t\x08\xc8\x01\x10\x80\x80\x80\x80\x02\x42\x02H\x03')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'sp_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'H\003'
  _TRAINERSPEC.fields_by_name['mining_sentence_size']._options = None
  _TRAINERSPEC.fields_by_name['mining_sentence_size']._serialized_options = b'\030\001'
  _TRAINERSPEC.fields_by_name['training_sentence_size']._options = None
  _TRAINERSPEC.fields_by_name['training_sentence_size']._serialized_options = b'\030\001'
  _TRAINERSPEC._serialized_start=28
  _TRAINERSPEC._serialized_end=1564
  _TRAINERSPEC_MODELTYPE._serialized_start=1500
  _TRAINERSPEC_MODELTYPE._serialized_end=1553
  _NORMALIZERSPEC._serialized_start=1567
  _NORMALIZERSPEC._serialized_end=1776
  _SELFTESTDATA._serialized_start=1778
  _SELFTESTDATA._serialized_end=1899
  _SELFTESTDATA_SAMPLE._serialized_start=1847
  _SELFTESTDATA_SAMPLE._serialized_end=1888
  _MODELPROTO._serialized_start=1902
  _MODELPROTO._serialized_end=2412
  _MODELPROTO_SENTENCEPIECE._serialized_start=2191
  _MODELPROTO_SENTENCEPIECE._serialized_end=2401
  _MODELPROTO_SENTENCEPIECE_TYPE._serialized_start=2306
  _MODELPROTO_SENTENCEPIECE_TYPE._serialized_end=2390
# @@protoc_insertion_point(module_scope)


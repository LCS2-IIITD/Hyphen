import os
import re
import logging
from   collections import OrderedDict
import xml.etree.ElementTree as ET
import penman
from   penman.models.noop import NoOpModel

logger = logging.getLogger(__name__)


# Cache for Penman graphs so they only need to be loaded once
pgraph_cache = {}
def load_amrs_cached(amr_fpath):
    global pgraph_cache
    pgraphs = pgraph_cache.get(amr_fpath, None)
    if pgraphs is None:
        pgraphs = penman.load(amr_fpath, model=NoOpModel())
        pgraph_cache[amr_fpath] = pgraphs
    return pgraphs


# Class for extracting Multi-sentence AMR data
class MSAMR(object):
    def __init__(self, xml_fpath):
        self.sents            = []
        self.sent_info        = {}
        self.ident_chain_dict = {}  # d[relationid] = list of dicts (type is key from collapsed tree tag)
        self.singleton_dict   = {}  # d[relationid] = list of dicts (type is key from collapsed tree tag)
        self.bridging_dict    = {}  # d[relationid] = list of dicts (type and sub-type are keys from collapsed tree tags)
        self._parse_xml(xml_fpath)  # fills in the above attribs

    def load_amrs(self, amr_fpath):
        pgraphs   = load_amrs_cached(amr_fpath)
        sent_ids  = self.get_sentence_ids()
        gdict     = {g.metadata['id']:g for g in pgraphs}
        odict     = OrderedDict()
        for sid in sent_ids:
            odict[sid] = gdict[sid]
        return odict

    def get_sentence_ids(self):
        sents = sorted(self.sents, key=lambda s:int(s['order']))
        sents = [s['id'] for s in sents]
        return sents

    def dump_corefs(self, identities=True, singletons=True, bridging=True):
        string  = ''
        string += 'Source : %s\n' % str(self.sent_info)
        string += '\n'.join(['  snum=%2d %s' % (i, str(s)) for i, s in enumerate(self.sents)]) + '\n'
        if identities:
            for key, vals in  self.ident_chain_dict.items():
                string += 'identity: %s\n' % key
                for val in vals:
                    string += '  ' + str(val) + '\n'
        if singletons:
            for key, vals in  self.singleton_dict.items():
                string += 'singleton: %s\n' % key
                for val in vals:
                    string += '  ' + str(val) + '\n'
        if bridging:
            for key, vals in  self.bridging_dict.items():
                string += 'bridging: %s\n' % key
                for val in vals:
                    string += '  ' + str(val) + '\n'
        return string

    def _parse_xml(self, fn):
        # root
        tree = ET.parse(fn)
        root = tree.getroot()
        assert root.tag == 'document'
        assert len(root) == 2
        sentences = root[0]
        relations = root[1]
        assert sentences.tag == 'sentences'
        assert relations.tag == 'relations'
        # Level 2 under sentences
        self.sent_info = sentences.attrib
        for sent in sentences:
            assert sent.tag == 'amr'
            self.sents.append(sent.attrib)
        # Level 2 under relations
        assert len(relations) == 3
        identity   = relations[0]
        singletons = relations[1]
        bridging   = relations[2]
        assert identity.tag   == 'identity'
        assert singletons.tag == 'singletons'
        assert bridging.tag   == 'bridging'
        # Level 3 under identity
        # These are mentions and implicity roles
        for identchain in identity:
            assert identchain.tag == 'identchain'
            key = identchain.attrib['relationid']
            assert key not in self.ident_chain_dict
            self.ident_chain_dict[key] = []
            for x in identchain:
                entry = {'type':x.tag, **x.attrib}
                self.ident_chain_dict[key].append( entry )
        # Level 3 under singletons
        # Sigletons are for identity chains that only participate in the bridging relations, not co-reference itself
        for identchain in singletons:
            assert identchain.tag == 'identchain'
            key = identchain.attrib['relationid']
            assert key not in self.singleton_dict
            self.singleton_dict[key] = []
            for x in identchain:
                entry = {'type':x.tag, **x.attrib}
                self.singleton_dict[key].append( entry )
        # Level 3 under bridging
        # Bridging relations are for co-references that are part of a set (ie.. 'they')
        for x in bridging:
            assert x.tag in ('setmember' 'partwhole')
            key = x.attrib['relationid']
            assert key not in self.bridging_dict
            self.bridging_dict[key] = []
            for y in x:
                entry = {'type':x.tag, 'subtype':y.tag, **y.attrib}
                self.bridging_dict[key].append(entry)


# Functions for building the file paths needed for loading multi-sentence data
# in LDC2020T02 (AMR3).  These are all for the "split" data.
class MSAMRFiles(object):
    name_re = re.compile(r'msamr_(\w+)_(\d+).xml$')
    def __init__(self, amr3_dir, is_train=False):
        self.amr3_dir = amr3_dir
        self.is_train = is_train
        # Get all the files in the directory
        tt_dir = 'train' if self.is_train else 'test'
        ms_dir = os.path.join(self.amr3_dir, 'data', 'multisentence', 'ms-amr-split', tt_dir)
        self.ms_fpaths = sorted([os.path.join(ms_dir, fn) for fn in os.listdir(ms_dir) if fn.startswith('msamr_')])

    # Get the file paths for all the multi-sentence xml files
    def get_ms_fpath(self, index):
        return self.ms_fpaths[index]

    # Get the short name for the xml file
    def get_name_number(self, index):
        ms_fpath = self.get_ms_fpath(index)
        match = self.name_re.search(ms_fpath)
        return match[1], match[2]

    # Get the test name
    def get_test_name(self, index):
        return '%s_%s' % self.get_name_number(index)

    # Get the standard (non-aligned) amr graph based on index of the xml file
    def get_amr_fpath(self, index):
        name, _ = self.get_name_number(index)
        tt_dir  = 'training' if self.is_train else 'test'
        fn      = 'amr-release-3.0-amrs-%s-%s.txt' % (tt_dir, name)
        fpath   = os.path.join(self.amr3_dir, 'data', 'amrs', 'split', tt_dir, fn)
        return fpath

    # Get the AMR graph with alignments based on the index of the xml file
    def get_amr_aligned_fpath(self, index):
        name, _ = self.get_name_number(index)
        tt_dir  = 'training' if self.is_train else 'test'
        fn      = 'amr-release-3.0-alignments-%s-%s.txt' % (tt_dir, name)
        fpath   = os.path.join(self.amr3_dir, 'data', 'alignments', 'split', tt_dir, fn)
        return fpath

    # Get the number fo files in in the directory to process
    def __len__(self):
        return len(self.ms_fpaths)

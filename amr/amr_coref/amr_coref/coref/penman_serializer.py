import logging
from   collections import Counter
import penman

logger = logging.getLogger(__name__)


# !!! Note that the penman graph must be decoded with model=NoOpModel(). When not using this,
# penman will by default, invert the x-of relations which revserses edges and causes some nodes
# to only have incoming arrows.  When you hit these, the serializer sees no children and fails
# to recurse the entire graph. penman.Graph doesn't own the model so at this point there's
# no way to check which model was used.  Checking here is done after serialization.
class PenmanSerializer(object):
    INSTANCE = ':instance'
    def __init__(self, pgraph):
        self.graph    = pgraph
        # Run the serialization
        self.elements = []              # clear elements list
        self.venum    = []              # enumerated variables (indexed same as elements)
        self.nodes    = set()           # nodes visited (to prevent recursion)
        self.serialize(pgraph.top)
        # Error check that all variables / nodes were recursed
        svars = set(v for v in self.venum if v != '_')
        if pgraph.variables() != svars:
            msg = 'Not all variables serialized (be sure you used penman NoOpModel): %s' % \
                (pgraph.metadata.get('id', None))
            assert False, msg
        # Error check that all attributes were serialized
        attrib_set = set(self.elements)
        for t in pgraph.attributes():
            attrib = t.target.replace('"', '').replace(' ', '_')    # must be same as below
            if attrib not in attrib_set:
                msg = 'Not all attributes serialized (be sure you used penman NoOpModel): %s' % \
                    (pgraph.metadata.get('id', None))
                assert False, msg

    # Return the string where variables are replaced by their values
    def get_graph_string(self):
        tokens = self.elements_to_tokens(self.elements)
        return ' '.join(tokens)

    # Return a string of the enumerated variables
    def get_variables_string(self):
        return ' '.join(self.venum)

    # Get the variable to concept map
    def get_var_to_concept(self):
        var2concept = {t.source:t.target for t in self.graph.instances()}
        return var2concept

    # Get the metadata from the graph
    def get_meta(self, key):
        return self.graph.metadata[key]

    # Depth first recurstion of the graph
    # Graph.triples are a list of (source, role, target)
    # note that in the recursive function, node_var is the "target" and could be an attribute/literal value
    def serialize(self, node_var):
        # If node_var is a variable and it's the first time we've seen it
        # Apply open paren if this is a variable (not an attrib/literal) and it's the first instance of it
        # If we've seen the variable before, don't insert a new node, just use the reference (ie.. no parens)
        if node_var in self.graph.variables() and node_var not in self.nodes:
            self.elements += [node_var]
            self.venum    += [node_var]
        # If it's a variable and it's the second+ time we've seen it
        elif node_var in self.graph.variables():
            self.elements += [node_var]
            self.venum    += [node_var]
            return
        # Otherwise it's an attribute
        else:
            attrib = node_var.replace('"', '').replace(' ', '_')
            self.elements += [attrib]
            self.venum    += ['_']
            return      # return if this isn't a variable or if it's the 2nd time we've see the variable
        self.nodes.add(node_var)
        # Loop through all the children of the node and recurse as needed
        children = [t for t in self.graph.triples if t[1] != self.INSTANCE and t[0] == node_var]
        for t in children:
            self.elements.append(t[1])      # add the edge, t[1] is role aka edge
            self.venum.append('_')
            self.serialize(t[2])            # recurse and add the child (node variable or attrib literal)

    # Convert the variables to concepts, but keep the roles and parens the same
    def elements_to_tokens(self, elements):
        # Get the mapping from variables to concepts and then update it (replace)
        # with any enumerated concepts
        var_dict = {t.source:t.target for t in self.graph.instances()}
        tokens   = [var_dict.get(x, x) for x in self.elements]
        return tokens

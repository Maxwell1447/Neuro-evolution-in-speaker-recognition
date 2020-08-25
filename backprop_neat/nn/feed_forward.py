from neat.graphs import feed_forward_layers
import torch


class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, torch.tensor(0.)) for key in inputs + outputs)

        self.input = torch.zeros(len(inputs))
        self.output = torch.zeros(len(outputs))

    def activate(self, inputs):
        # INPUTS   batch_size x BIN
        if len(self.input_nodes) != inputs.shape[-1]:
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs.transpose(0, 1)):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = torch.empty(len(links), inputs.shape[0])
            idx = 0
            for i, w in links:
                node_inputs[idx] = self.values[i] * w
                idx += 1

            s = agg_func(node_inputs)

            self.values[node] = act_func(bias + response * s)

        out = torch.empty(len(self.output_nodes), inputs.shape[0])
        for i in self.output_nodes:
            out[i] = self.values[i]
        return out.t()

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        # TODO: check tensor
                        assert isinstance(cg.weight, torch.Tensor) and cg.weight.requires_grad
                        inputs.append((inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                # TODO: check tensor
                assert isinstance(ng.bias, torch.Tensor) and ng.bias.requires_grad
                node_evals.append((node, activation_function, aggregation_function, ng.bias, torch.tensor(1.), inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)


class FeedForwardNet(object):
    def __init__(self, key_in, key_out, layers_conn, layers_node):
        self.key_in = key_in
        self.key_out = key_out
        self.layers_conn = layers_conn
        self.layers_node = layers_node
        self.recreate = True

        self.input = torch.zeros(len(key_in))
        self.output = torch.zeros(len(key_out))

    def activate(self, inputs):

        self.init_connection_matrices()

        # INPUTS   batch_size x BIN
        if len(self.key_in) != inputs.shape[-1]:
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.key_in), len(inputs)))

        for k, v in zip(self.key_in, inputs.transpose(0, 1)):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = torch.empty(len(links), inputs.shape[0])
            idx = 0
            for i, w in links:
                node_inputs[idx] = self.values[i] * w
                idx += 1

            s = agg_func(node_inputs)

            self.values[node] = act_func(bias + response * s)

        out = torch.empty(len(self.output_nodes), inputs.shape[0])
        for i in self.output_nodes:
            out[i] = self.values[i]
        return out.t()

    def init_connection_matrices(self):
        ...

    def train(self):
        self.recreate = True

    def eval(self):
        self.recreate = False
        self.init_connection_matrices()


    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        smart_connections = {}
        for cg_in, cg_out in connections:
            if cg_out in smart_connections:
                smart_connections[cg_out].append(cg_in)
            else:
                smart_connections[cg_out] = [cg_in]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)

        node_to_layer = {}
        for node in config.genome_config.input_keys:
            node_to_layer[node] = 0
        for i, layer in enumerate(layers):
            for node in layer:
                node_to_layer[node] = i + 1

        layers_conn = []
        layers_node = []
        for i, layer in enumerate(layers):
            layer_conn = []  # (layer in, node in, node out, weight)
            layer_node = []  # (node, activation, aggregation, bias)
            for node in layer:
                for inode in smart_connections[node]:
                    cg = genome.connections[(inode, node)]
                    layer_conn.append((node_to_layer[inode], inode, node, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)

                layer_node.append((node, activation_function, aggregation_function, ng.bias))
            layers_conn.append(layer_conn)
            layers_node.append(layer_node)

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys,
                                  layers_conn, layer_node)

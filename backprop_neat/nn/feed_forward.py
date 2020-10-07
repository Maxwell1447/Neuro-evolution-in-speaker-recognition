from neat.graphs import feed_forward_layers
import torch
from neat_local.pytorch_neat.activations import sigmoid_activation


def dense_from_coo(shape, conns, mapping, dtype=torch.float64, device="cpu"):
    mat = torch.zeros(shape, dtype=dtype, device=device)

    for key in conns:
        # assert isinstance(weight, torch.Tensor) and weight.requires_grad
        key_in, key_out = key
        row = mapping[key_in][0]
        col = mapping[key_out][1]
        mat[row, col] = conns[key]
    return mat


def dense_from_node(shape, nodes, mapping, dtype=torch.float64, device="cpu"):
    mat = torch.empty(shape, dtype=dtype, device=device)
    
    for key in nodes:
        mat[mapping[key][1]] = nodes[key]
    
    return mat


class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, connection_layers, node_layers,
                 mapping,
                 activation=sigmoid_activation,
                 dtype=torch.float64,
                 device="cpu"):
        self.input_nodes = inputs

        self.output_nodes = outputs
        self.layers = list(map(len, node_layers))
        # print("mapping", mapping)
        # print("layers", self.layers)
        self.dtype = dtype
        self.device = device
        self.activation = activation

        self.layer_matrices = []
        self.layer_biases = []

        for i in range(len(self.layers)):
            self.layer_matrices.append(dense_from_coo((len(inputs), self.layers[i]),
                                                      connection_layers[i], mapping))
            self.layer_biases.append(dense_from_node(self.layers[i],
                                                      node_layers[i], mapping))
            inputs += list(node_layers[i])

        self.input = torch.zeros(len(inputs))
        self.output = torch.zeros(len(outputs))

    def activate(self, inputs):
        """
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        """

        current_inputs = inputs

        num_layers = len(self.layers)

        for i, matrix in enumerate(self.layer_matrices):
            activs = self.activation(
                current_inputs.mm(matrix)
                + self.layer_biases[i]
            )

            if i + 1 == num_layers:
                return activs

            current_inputs = torch.cat((current_inputs, activs), dim=1)

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)

        connection_layers = []
        node_layers = []
        mapping = {}
        global_idx = len(config.genome_config.input_keys)

        for i, input_key in enumerate(config.genome_config.input_keys):
            mapping[input_key] = (i, i)

        layer_num = 0
        for layer in layers:
            local_idx = 0
            connection_layers.append({})
            node_layers.append({})
            for node in layer:
                mapping[node] = (global_idx, local_idx)
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        # TODO: check tensor
                        # assert isinstance(cg.weight, torch.Tensor) and cg.weight.requires_grad
                        connection_layers[layer_num][conn_key] = cg.weight
                local_idx += 1
                global_idx += 1

                ng = genome.nodes[node]

                # TODO: check tensor
                # assert isinstance(ng.bias, torch.Tensor) and ng.bias.requires_grad
                node_layers[layer_num][node] = ng.bias
            layer_num += 1

        return FeedForwardNetwork(config.genome_config.input_keys.copy(), config.genome_config.output_keys,
                                  connection_layers, node_layers, mapping)


import torch

# different rules to update the weight

def basic_update(self, inputs, w):
    d_ws = torch.zeros(inputs.size(0))
    for idx, x in enumerate(inputs):
        y = torch.dot(w, x)
        d_w = torch.zeros(w.shape)
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                d_w[i, j] = self.c * x[j] * y[i]
        d_ws[idx] = d_w

    return torch.mean(d_ws, dim=0)


def oja_update(self, inputs, w):
    d_ws = torch.zeros(inputs.size(0), *w.shape)
    for idx, x in enumerate(inputs):
        y = torch.mm(w, x.unsqueeze(1))
        d_w = torch.zeros(w.shape)
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                d_w[i, j] = self.c * y[i] * (x[j] - y[i] * w[i, j])
        d_ws[idx] = d_w

    return torch.mean(d_ws, dim=0)

def krotov_update(self, inputs, weights):
        batch_size = inputs.shape[0]
        num_hidden_units = weights.shape[0]
        input_size = inputs[0].shape[0]
        assert (self.k <= num_hidden_units), "The amount of hidden units should be larger or equal to k!"
        if self.normalize:
            norm = torch.norm(inputs, dim=1)
            norm[norm == 0] = 1
            inputs = torch.div(inputs, norm.view(-1, 1))

        inputs = torch.t(inputs)

        # Calculate overlap for each hidden unit and input sample
        tot_input = torch.matmul(torch.sign(weights) * torch.abs(weights) ** (self.norm - 1), inputs)

        # Get the top k activations for each input sample (hidden units ranked per input sample)
        _, indices = torch.topk(tot_input, k=self.k, dim=0)

        # Apply the activation function for each input sample
        activations = torch.zeros((num_hidden_units, batch_size))
        activations[indices[0], torch.arange(batch_size)] = 1.0
        activations[indices[self.k - 1], torch.arange(batch_size)] = -self.delta

        # Sum the activations for each hidden unit, the batch dimension is removed here
        xx = torch.sum(torch.mul(activations, tot_input), 1)

        # Apply the actual learning rule, from here on the tensor has the same dimension as the weights
        norm_factor = torch.mul(xx.view(xx.shape[0], 1).repeat((1, input_size)), weights)
        ds = torch.matmul(activations, torch.t(inputs)) - norm_factor

        # Normalize the weight updates so that the largest update is 1 (which is then multiplied by the learning rate)
        nc = torch.max(torch.abs(ds))
        if nc < self.precision:
            nc = self.precision
        d_w = torch.true_divide(ds, nc)

        return d_w


import torch

# EMA of parameter deltas
class DeltaEMA:

    """
        Initializes the DeltaEMA tracker

        Args:
            beta (float): Smoothing factor for the EMA (default: 0.9).
                          Closer to 1.0 means slower EMA updates.
        """

    def __init__(self, beta=0.9):
        self.beta = beta  # Smoothing factor for EMA (closer to 1 = slower updates)
        self.prev = {}  # Stores previous value of each parameter (by id)
        self.ema = 0.0  # The current EMA value

    @torch.no_grad()
    def update(self, params):
        """
        Updates the EMA with the mean absolute change of each parameter.

        Args:
            params (list): List of torch.nn.Parameter to track.

        Returns:
            float: The current EMA value after the update.
        """
        # If no parameters provided, just return the current EMA

        if not params:
            return float(self.ema)
        # we want to store the mean absolute changes (magnitudes)
        mags = []
        for p in params:
            pid = id(p)
            cur = p.data.detach().cpu()  # copy the current value (new value)

            # If this parameter is new store its initial value and skip delta computation
            if pid not in self.prev:
                self.prev[pid] = cur.clone()
                continue

            # Compute mean absolute change since last step (current - previous) save the mean absolute change in the list
            mags.append((cur - self.prev[pid]).abs().mean().item())

            # Update stored previous value to current value
            self.prev[pid] = cur.clone()  # store the current value in previous for later comparison to become the (old value)
        if mags:
            # Compute average delta step (current change mean)
            step = sum(mags) / len(mags)
            self.ema = self.beta * self.ema + (1 - self.beta) * step
        return float(self.ema)




# 2) Collect parameter groups: W (scale_base/scale_sp) vs C (coef)

# seen :prevents duplicates in the optimizer setup.
# when the training loop starts, all parameters in the optimizer still get updated.

def collect_kan_param_groups(model):
    W_params, C_params, seen = [], [], set()
    for m in model.modules():
        if isinstance(m, KANLayer):
            for name in ("scale_base", "scale_sp"):
                p = getattr(m, name, None) # get an attribute from model
                if isinstance(p, torch.nn.Parameter) and id(p) not in seen: # Check if p is a torch.nn.Parameter object
                    W_params.append(p); seen.add(id(p))
            p = getattr(m, "coef", None)
            if isinstance(p, torch.nn.Parameter) and id(p) not in seen:
                C_params.append(p); seen.add(id(p))
    # for p in model.parameters():
    #     if id(p) not in seen:
    #         W_params.append(p); seen.add(id(p))

    return W_params, C_params # returns two disjoint lists we can give two different learning rates

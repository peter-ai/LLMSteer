import torch
import torch.nn as nn


# ----------------------- Regular Models (No-Norm) ----------------------- #
class ER_2H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_3H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=75),
            nn.ReLU(),
            nn.Linear(in_features=75, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_3H_v2_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_3H_v3_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_4H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_4H_v2_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=750),
            nn.ReLU(),
            nn.Linear(in_features=750, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=450),
            nn.ReLU(),
            nn.Linear(in_features=450, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_5H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_5H_v2_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_6H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_6H_v2_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_7H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_7H_v2_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=2500),
            nn.ReLU(),
            nn.Linear(in_features=2500, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


# ----------------------- LayerNorm Models ----------------------- #
class ER_2H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=25),
            nn.LayerNorm(25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_3H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=75),
            nn.LayerNorm(75),
            nn.ReLU(),
            nn.Linear(in_features=75, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_3H_v2_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_3H_v3_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=150),
            nn.LayerNorm(150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_4H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=500),
            nn.LayerNorm(500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_4H_v2_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=750),
            nn.LayerNorm(750),
            nn.ReLU(),
            nn.Linear(in_features=750, out_features=600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=450),
            nn.LayerNorm(450),
            nn.ReLU(),
            nn.Linear(in_features=450, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_5H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_5H_v2_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=1200),
            nn.LayerNorm(1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

class ER_6H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=1500),
            nn.LayerNorm(1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1200),
            nn.LayerNorm(1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_6H_v2_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=2000),
            nn.LayerNorm(2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1500),
            nn.LayerNorm(1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_7H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=2000),
            nn.LayerNorm(2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1500),
            nn.LayerNorm(1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1200),
            nn.LayerNorm(1200),
            nn.ReLU(),
            nn.Linear(in_features=1200, out_features=1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs


class ER_7H_v2_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=2500),
            nn.LayerNorm(2500),
            nn.ReLU(),
            nn.Linear(in_features=2500, out_features=2000),
            nn.LayerNorm(2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1500),
            nn.LayerNorm(1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1000),
            nn.LayerNorm(1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.LayerNorm(800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=600),
            nn.LayerNorm(600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=49),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs
    

# ----------------------- Loss Functions ----------------------- #
class ERLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(ERLoss, self).__init__(*args, **kwargs)

    def forward(self, pred_probs: torch.Tensor, hint_latencies : torch.Tensor, opt_latency: torch.Tensor):
        expected_regret = torch.sum(pred_probs * (hint_latencies - opt_latency), dim=hint_latencies.dim()-1)
        return expected_regret.mean()
    

# ----------------------- Model Exports ----------------------- #
ER_MODELS = [
    ER_2H_v1_Model,
    ER_3H_v1_Model,
    ER_3H_v2_Model,
    ER_3H_v3_Model,
    ER_4H_v1_Model,
    ER_4H_v2_Model,
    ER_5H_v1_Model,
    ER_5H_v2_Model,
    ER_6H_v1_Model,
    ER_6H_v2_Model,
    ER_7H_v1_Model,
    ER_7H_v2_Model,
]

ER_LN_MODELS = [
    ER_2H_v1_LN_Model,
    ER_3H_v1_LN_Model,
    ER_3H_v2_LN_Model,
    ER_3H_v3_LN_Model,
    ER_4H_v1_LN_Model,
    ER_4H_v2_LN_Model,
    ER_5H_v1_LN_Model,
    ER_5H_v2_LN_Model,
    ER_6H_v1_LN_Model,
    ER_6H_v2_LN_Model,
    ER_7H_v1_LN_Model,
    ER_7H_v2_LN_Model,
]

MULTICLASS_RUN_MODES = {
    'EROnly': [ *ER_MODELS, *ER_LN_MODELS ], 
    'BCEOnly': [],
    'LNOnly': [ *ER_LN_MODELS, ], 
    'NLNOnly': [ *ER_MODELS, ], 
    'ALL': [ *ER_MODELS, *ER_LN_MODELS, ],
}
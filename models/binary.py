import torch.nn as nn


# ----------------------- Regular Models (No-Norm) ----------------------- #
class BCE_2H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_3H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=75),
            nn.ReLU(),
            nn.Linear(in_features=75, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_3H_v2_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_3H_v3_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_4H_v1_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_4H_v2_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=750),
            nn.ReLU(),
            nn.Linear(in_features=750, out_features=600),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=450),
            nn.ReLU(),
            nn.Linear(in_features=450, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_5H_v1_Model(nn.Module):
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
            nn.Linear(in_features=400, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_5H_v2_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_6H_v1_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_6H_v2_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_7H_v1_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_7H_v2_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


# ----------------------- LayerNorm Models ----------------------- #
class BCE_2H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=25),
            nn.LayerNorm(25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_3H_v1_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=75),
            nn.LayerNorm(75),
            nn.ReLU(),
            nn.Linear(in_features=75, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_3H_v2_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=200),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=100),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_3H_v3_LN_Model(nn.Module):
    def __init__(self, inputs: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            nn.Linear(in_features=inputs, out_features=300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=150),
            nn.LayerNorm(150),
            nn.ReLU(),
            nn.Linear(in_features=150, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_4H_v1_LN_Model(nn.Module):
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
            nn.Linear(in_features=300, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_4H_v2_LN_Model(nn.Module):
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
            nn.Linear(in_features=450, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_5H_v1_LN_Model(nn.Module):
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
            nn.Linear(in_features=400, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_5H_v2_LN_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_6H_v1_LN_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_6H_v2_LN_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


class BCE_7H_v1_LN_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()
    

class BCE_7H_v2_LN_Model(nn.Module):
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
            nn.Linear(in_features=600, out_features=1),
        )
    
    def forward(self, x):
        probs = self.model(x)
        return probs.flatten()


# ----------------------- Model Exports ----------------------- #
BCE_MODELS = [
    BCE_2H_v1_Model,
    BCE_3H_v1_Model,
    BCE_3H_v2_Model,
    BCE_3H_v3_Model,
    BCE_4H_v1_Model,
    BCE_4H_v2_Model,
    # BCE_5H_v1_Model,
    # BCE_5H_v2_Model,
    # BCE_6H_v1_Model,
    # BCE_6H_v2_Model,
    # BCE_7H_v1_Model,
    # BCE_7H_v2_Model,
]
BCE_LN_MODELS = [
    BCE_2H_v1_LN_Model,
    BCE_3H_v1_LN_Model,
    BCE_3H_v2_LN_Model,
    BCE_3H_v3_LN_Model,
    BCE_4H_v1_LN_Model,
    BCE_4H_v2_LN_Model,
    # BCE_5H_v1_LN_Model,
    # BCE_5H_v2_LN_Model,
    # BCE_6H_v1_LN_Model,
    # BCE_6H_v2_LN_Model,
    # BCE_7H_v1_LN_Model,
    # BCE_7H_v2_LN_Model,
]

BINARY_RUN_MODES = {
    'EROnly': [],
    'BCEOnly': [ *BCE_MODELS, *BCE_LN_MODELS ],
    'LNOnly': [ *BCE_LN_MODELS ], 
    'NLNOnly': [ *BCE_MODELS ], 
    'ALL': [ *BCE_MODELS, *BCE_LN_MODELS ],
}
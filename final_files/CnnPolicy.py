import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 512):
        super().__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            cnn_output_dim = self.cnn(sample).shape[1]

        self.linear = nn.Linear(cnn_output_dim, features_dim)

    def forward(self, obs):
        return self.linear(self.cnn(obs))

# cheat sheet below 
"""
la intrare (3, 20, 20) - 3 canale de tabla 20x20

features_dim = 256 - nici prea mic nici prea mare - cate numere in vector sa fie date catre ppo la final
-----------------------------------------------------------------------------
Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)

3 canale la intrare, x (64, 128) la iesire
64/128 de filtre de dimensiune 3x3 + canale iese practic 3x3x3
in stratul 2 ajung filtrele unlee langa altele - daca se fac abstractie de duplicate
    se ajunge la un spatiu de 5x5x5
filtrele sunt initializaat random si se ajusteaza prin backprop

3. PADDING ȘI MATEMATICA DIMENSIUNII
-----------------------------------------------------------------------------
- Scopul padding=1: "SAME Padding". Păstrează tabla de 20x20 intactă.
- De ce? Centrul filtrului de 3x3 are nevoie de 1 strat de zerouri exterioare 
  pentru a putea citi colțurile/marginile extreme ale tablei.
- Formula generală: Padding = (Kernel_Size - 1) / 2
  (ex: pentru kernel 5x5, padding-ul ar fi 2).

4. CÂMPUL RECEPTIV (Efectul de Piramidă: 3x3 -> 5x5 -> 7x7)
-----------------------------------------------------------------------------
- Stratul 2 are tot filtre 3x3, dar se uită la hărțile din Stratul 1.
- Deoarece fiecare celulă din Stratul 1 "vede" o zonă 3x3 din tabla originală, 
  când Stratul 2 combină 3 celule adiacente, aria vizualizată se suprapune.
- Regula: Fiecare strat convoluțional de 3x3 adaugă +2 la "viziunea" globală.
  -> Conv1 = vede 3x3 pe tablă
  -> Conv2 = vede 5x5 pe tablă
  -> Conv3 = vede 7x7 pe tablă

  stride =2 sare peste un patratel - un fel de zoom out - ajuta reteaua sa detecteze structuri mai mari

5. FLATTEN ȘI LINEAR (Trecerea de la spațial la dens)
-----------------------------------------------------------------------------
- Flatten(): Ia toate hărțile de 20x20 și le "pisează" într-un vector uriaș 
  1D (ex: 51200 numere). Acest vector conține "inventarul" întregii table.
- Linear(51200, 256): Comprimă acel inventar zgomotos în cele mai importante 
  256 de concepte esențiale (features_dim).

6. MASKABLE PPO / ACTOR-CRITIC (Forma de "Y")
-----------------------------------------------------------------------------
Cele 256 de numere extrase de tine intră în algoritmul PPO, care are 2 "capete":
- ACTORUL (Decidentul): Scoate un vector de probabilități pentru fiecare mutare 
  (ex: 400 de mutări). 
  -> Action Masking: Taie la 0% mutările ilegale (ocupate) înainte de a muta.
- CRITICUL (Analistul): Scoate UN SINGUR NUMĂR (Valoarea Stării).
  -> Estimează probabilitatea globală de a câștiga jocul de la starea actuală 
     (ex: de la -1.0 la +1.0).

7. `torch.no_grad()` ȘI DUMMY TENSOR
-----------------------------------------------------------------------------
- Folosit DOAR în faza de __init__ (inițializare).
- `dummy_tensor = torch.zeros(1, 3, 20, 20)`: O tablă "falsă" pe care o trecem 
  prin CNN doar pentru a "măsura" automat câți pixeli ies înainte de Flatten 
  (ex: 51200), ca să construim corect stratul Linear, fără a calcula manual.
- `torch.no_grad()`: Îi spune lui PyTorch să "nu învețe nimic" (să nu calculeze 
  gradienți) din acest tensor fals, economisind astfel timp și memorie.
- În timpul antrenamentului (la joc), no_grad() nu este folosit, iar rețeaua 
  CNN învață constant pe baza gradienților de la PPO.

  8. FUNCȚIA `forward(self, obs)` (Fluxul de execuție)
-----------------------------------------------------------------------------
- Ce face: Definește traseul informației (ordinea de execuție) în timp real.
- Funcționează din interior spre exterior: `return self.linear(self.cnn(obs))`
  1) `obs` -> intră tensorul tablei (ex: stare de joc live).
  2) `self.cnn(obs)` -> se aplică filtrele și Flatten-ul = 51200 de numere.
  3) `self.linear(...)` -> face compresia finală.
  4) `return` -> scuipă cele 256 de features.
"""
# FE Stats Classifier
---
### Stats to Analyze
- HP
- Atk (If not Echoes or Binding Blade, take max between Str & Mag)
- Skl/Dex
- Spd
- Lck
- Def
- Res
---
---
### Metrics
- Mean = sum of stat / count of stat
  - ### $`\mu={\frac {1}{n}}\sum _{i=1}^{n}x_{i}`$
- Standard Deviation = sqrt(sum of diff of mean squared / count of stat)
  - ### $`\sigma = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \mu)^2}`$
- Standardization = diff of mean / std
  - ### $`z=\frac{x - \mu}{\sigma}`$
---
---
### Key Points
- Analysis based on comparison with character with most amount of appearances (provided stats are comparable) and stat disribution (in %) after normalizing data and averaging
- Use base stats from joining time
- Include DLC and other means outside of strict base game availability
- No spin-offs (except FEH)
- Do not include seasonal FEH alts
---
---
### Characters
**Marth (13)**
- Main line [1,3,11,12,13 as DLC 2x,14,15]
- FEH [Base, Legend, Youth, Brave, Plot Ghost]

**Linde (7)**
- Main line [1,3,11,12,13 as DLC]
- FEH [Base, Khadein]

**Palla (10)**
- Main line [1,2,3,11,12,13 as DLC,15]
- FEH [Base, Youth, Falcon Knight]

**Alm (6)**
- Main line [2,13 as DLC,15]
- FEH [Base, Legend, Brave]

**Roy (6)**
- Main line [6,13 as DLC,15]
- FEH [Base, Legend, Brave]

**Lilina (5)**
- Main line [6,13 as DLC]
- FEH [Base, Legend, Rearmed]

**Claude (5)**
- Main line [16 + DLC]
- FEH [Base, Brave, Legend]

**Lysithea (3)**
- Main line [16]
- FEH [Base, Brave]
---
---
### Model Algorithms Used
- Vanilla MLP (Multi-Layered Perceptron)
- CNN (Convolutional Neural Network)
- SVM (Support Vector Machine)
- Random Forest
---
---
### Model Algorithms Excluded
- RNN (Recurrent Neural Network, not temporal data)
- LSTM (Long Short-Term Memory, no need for memory)
- Transformers (Not efficient)
- Linear Regression (Not real number values)
- Logistic Regression (Not binary classification)
- Decision Tree (Not sophisticated enough)
- Naive Bayes (Not sophisticated enough)
---
---
### Data Sources
- [Serenes Forest](https://serenesforest.net)
  - [FE1 Stats](https://serenesforest.net/shadow-dragon-and-blade-of-light/characters/base-stats/)
  - [FE2 Stats](https://serenesforest.net/gaiden/characters/base-stats/)
  - [FE3 Stats](https://serenesforest.net/mystery-of-the-emblem/characters/base-stats/)
  - [FE6 Stats](https://serenesforest.net/binding-blade/characters/base-stats/)
  - [FE11 Stats](https://serenesforest.net/shadow-dragon/characters/base-stats/default/)
  - [FE12 Stats](https://serenesforest.net/light-and-shadow/characters/base-stats/default/)
  - [FE13 Stats (DLC)](https://serenesforest.net/awakening/characters/base-stats/dlc/)
  - [FE13 Stats (Spot Pass)](https://serenesforest.net/awakening/characters/base-stats/spotpass/)
  - [FE14 Stats](https://serenesforest.net/fire-emblem-fates/other-characters/base-stats/)
  - [FE15 Stats](https://serenesforest.net/fire-emblem-echoes-shadows-valentia/characters/base-stats/)
  - [FE15 Stats](https://serenesforest.net/fire-emblem-echoes-shadows-valentia/miscellaneous/amiibo/)
  - [FE16 Stats](https://serenesforest.net/three-houses/characters/base-stats/)
- [Fire Emblem Heroes Wiki (Fandom)](https://feheroes.fandom.com/wiki/List_of_Heroes) Hero List
---
---
### Legal
- This project is a non-profit, informational data analysis experiment. *Fire Emblem* and all its related material are copyrighted by Nintendo/Intelligent Systems. This project is in no way affiliated or related to either of the companies.
---
---

# Original Notebook Dataset

The [course notebooks](../notebooks/) use `ufc-master.csv`. These notes describe that original input. The website now loads a separate, versioned set of fighter records and model files.

## Columns

| Fields | Contents |
|--------|----------|
| `RedFighter`, `BlueFighter` | Fighter names |
| `RedWins`, `BlueWins`, `RedLosses`, `BlueLosses` | Recorded career wins and losses |
| `RedHeightCms`, `BlueHeightCms`, `RedReachCms`, `BlueReachCms` | Height and reach in centimeters |
| `RedAvgSigStrLanded`, `BlueAvgSigStrLanded` | Significant strikes landed |
| `RedAvgTDLanded`, `BlueAvgTDLanded` | Takedowns landed |
| `RedAvgSubAtt`, `BlueAvgSubAtt` | Submission attempts |
| `RedStance`, `BlueStance` | Stance |
| `Date`, `Location`, `WeightClass`, `TitleBout` | Bout conditions |
| `Winner` | Red or blue corner |
| Columns ending in `Dif` | Precomputed differences between fighters |

The notebook data-loading cells contain paths from the original Colab setup. Update them to the location of your CSV before running. The runtime download described in the [README](../README.md#quick-start) includes `data/ufc-master.csv`.

Notebook results belong to the original analysis. The deployed model's current evaluation is on the [results page](https://versus.thomasou.com/results).

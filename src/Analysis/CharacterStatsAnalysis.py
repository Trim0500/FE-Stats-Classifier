import matplotlib.pyplot as plt
from math import ceil
from numpy import array
from pandas import DataFrame
from torch import tensor, sqrt, sort, abs
from StatsTableSingleton import StatsTableSingleton

class CharacterStatsAnalysis():
    """
        Class that abstracts the statistical and analytical procedures for the character stat line.

        Class Members: 
            stats_names: List representing the stat labels to use in plots
            stats_colors: List representing the stat label colors to use in plots
        
        Instance Members: 
            tensor_device: String representing the hardware device to run tensors on
            character_name: String representing the character name for filtering
            index_labels: List representing the game entries for the character
            stats_table: StatsTableSingleton instance representing the "database" connection to retrieve data
            filtered_stats_dataframe: DataFrame instance representing the complete stat line table for the character
            stats: Tensor instance representing the complete stat line floating point data from the table
            normalized_stats: Tensor instance representing the normalized stat lines
            ave_stats: Tensor instance representing the normalized average stat line
        
    """

    stats_names = ["HP","Atk","Skl","Spd","Lck","Def","Res"]
    stats_colors = ["darkorange","darkred","darkblue","darkgreen","gold","#cccc00","royalblue"]

    def __init__(_self, _tensor_device: str, _character_name: str, _index_labels: list) -> None:
        _self.tensor_device = _tensor_device
        _self.character_name = _character_name
        _self.index_labels = _index_labels
        _self.stats_table = StatsTableSingleton()
        _self.filtered_stats_dataframe = DataFrame([])
        _self.stats = tensor([])
        _self.normalized_stats = tensor([])
        _self.ave_stats = tensor([])

    def filter_dataframe(_self) -> None:
        """
            Method to use the singleton instance to get the filtered dataframe and convert to tensor.

            Raises:
                exc: Exception
        """
        try:
            _self.filtered_stats_dataframe = _self.stats_table.get_stats_by_name(_self.character_name, _self.index_labels)
            _self.stats = tensor(_self.filtered_stats_dataframe.to_numpy(), dtype=float, device=_self.tensor_device)
        except Exception as exc:
            print(f"[CharacterStatsAnalysis/filter_dataframe]: {str(exc)}")

    def normalize_stats(_self) -> None:
        """
            Method to utilize the stats of a character across game appearences to determine normalized stats.
            Note: Some game appearences don't always have values for specific stats, so mean and standard deviation is affected.

            Raises:
                exc: Exception
        """
        try:
            """
                Run a check first to see if we even should normalize. 
                Get the mean of the each stat first, if about 80% of the values don't go beyond a tolerance threashold of 2.0, just average up the stats instead.
                Note: Watch for sentinel values.
            """

            sentinel_mask = _self.stats != -99
            masked_stats_tensor = _self.stats * sentinel_mask
            mean_tensor = (masked_stats_tensor).sum(dim=0) / sentinel_mask.sum(dim=0)
            threshold_mask = (abs(masked_stats_tensor - mean_tensor) <= 2.0) | (abs(masked_stats_tensor - mean_tensor) == mean_tensor)
            if threshold_mask.sum() >= round((_self.stats.shape[0] * _self.stats.shape[1]) * 0.8, 0):
                _self.normalized_stats = _self.stats
                
                _self.ave_stats = mean_tensor

                return

            std_list = []

            for i in range(len(mean_tensor)):
                stat_tensor = _self.stats[:,i]
                stat_tensor = stat_tensor[stat_tensor != -99]
                std_list.append(sqrt(((stat_tensor - mean_tensor[i])**2).sum() / len(stat_tensor)))

            std_tensor = tensor(std_list, device=_self.tensor_device)
            _self.normalized_stats = _self.stats - mean_tensor / std_tensor
            _self.ave_stats = (_self.normalized_stats * sentinel_mask).sum(dim=0) / sentinel_mask.sum(dim=0)
        except Exception as exc:
            print(f"[CharacterStatsAnalysis/normalize_stats]: {str(exc)}")

    def plot_ave_stat_distribution(_self) -> None:
        """
            Method to visualize the normalized average stat line for a particular character through a pie plot.

            Raises:
                exc: Exception
        """
        try:
            plt.figure(figsize=(9,9), facecolor="lightgrey")

            ave_stats, indices = sort(_self.ave_stats, descending=True)

            names = array(CharacterStatsAnalysis.stats_names)
            names = names[indices.cpu()]
            
            colors = array(CharacterStatsAnalysis.stats_colors)
            colors = colors[indices.cpu()]

            patches, texts, pcts = plt.pie(abs(ave_stats).cpu(),
                                            counterclock=False,
                                            startangle=90,
                                            colors=colors,
                                            labels=names,
                                            autopct="%1.1f%%",
                                            textprops={ "size": "larger", "fontweight": "bold" },
                                            wedgeprops={ "linewidth": 1.5, "edgecolor": "white" },
                                            shadow={'ox': -0.02, 'edgecolor': 'none', 'shade': 0.6})
            plt.setp(pcts, color="white")

            plt.title(f"Average Normalized Stat Distribution for {_self.character_name}", fontdict={ "fontsize": "x-large", "fontweight": "bold" }, loc="left")

            plt.show()
        except Exception as exc:
            print(f"[FE_Stats_Data_Analysis/plot_ave_stat_distribution]: {str(exc)}")

    def plot_all_percentage_stat_lines(_self) -> None:
        """
            Method to visualize all normalized stat lines for a particular character through a pie plot.

            Raises:
                exc: Exception
        """
        try:
            fig, axes = plt.subplots(nrows=ceil(_self.stats.shape[0] / 4), ncols=4, figsize=(12,12), layout="constrained", facecolor="lightgrey")

            for i, ax in enumerate(axes.flat):
                if i >= len(_self.stats):
                    break

                buffer = _self.stats[i].cpu().to(int)

                name_buffer = array(CharacterStatsAnalysis.stats_names)

                color_buffer = array(CharacterStatsAnalysis.stats_colors)

                if ((buffer == -99) | (buffer == 0)).sum() > 0:
                    match_indices = ((buffer != -99) & (buffer != 0)).nonzero().view(-1)
                    buffer = buffer[match_indices.numpy()]

                    name_buffer = name_buffer[match_indices.numpy()]
                    
                    color_buffer = color_buffer[match_indices.numpy()]
                
                buffer, indices = sort(buffer, descending=True)

                name_buffer = name_buffer[indices]
                
                color_buffer = color_buffer[indices]

                patches, texts, pcts = ax.pie(buffer,
                                                counterclock=False,
                                                startangle=90,
                                                colors=color_buffer,
                                                labels=name_buffer,
                                                autopct="%1.1f%%",
                                                textprops={ "size": "small", "fontweight": "bold" },
                                                wedgeprops={ "linewidth": 1.5, "edgecolor": "white" },
                                                shadow={'ox': -0.02, 'edgecolor': 'none', 'shade': 0.6})
                plt.setp(pcts, color="white")

                ax.set_title(f"{_self.character_name} {_self.index_labels[i]} Stat Dist.", fontdict={ "fontsize": "small", "fontweight": "bold" }, loc="left")

            plt.show()
        except Exception as exc:
            print(f"[FE_Stats_Data_Analysis/process_percentage_tensor]: {str(exc)}")


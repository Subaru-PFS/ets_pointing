# PPP online tool

This tool is designed for users preparing the PFS open-use proposals. It could help users to:

* estimate the total required hours / fiber hours;
* know the expected completeness of the user sample as a function of PPC (PFS pointing centers);
* evaluate the fiber usage fraction of each PPC.

## Getting Started

### Prerequisites

* Python 3
* The following packages are required:
```
pip install astropy seaborn colorcet psutil scikit-learn
```
* netflow 
    * refer to [here](https://github.com/Subaru-PFS/ets_fiberalloc)

### Executing program
* make a target list including "obj_id", "ra", "dec", "equinox", "priority", "exptime", "resolution" & "comment"
    * "ra" & "dec" should be given in unit of degree or hms,dms
    * "priority" is the inner priority defined by users, it should be gievn as an integer between 0 to 9, with 0 meaning the highest priority
    * "exp_time" is the required observation time for each target, it should be given in unit of seconds
    * "resolution" can be 'L' (low-r) or 'M' (medium-r), default is 'L'
    * the format can be csv, ecsv or fits
    * astropy.Table can easily generate the file, refer to [here](https://github.com/Subaru-PFS/ets_target_database/blob/main/docs/input_target_list.md)

* save file in "input/" 
* run the script in Jupyter

### outputs

* total time required 
	* the time would be given in unit of hours and fiber hours
	* if your sample requires both the low- and medium-resolution modes, it would print outputs in the respective mode separately ~~you need to sum up the outputs in the respective mode to derive the total required hours / fiber hours~~ (Jun/6/23)
    * both the total on-source time and required time (including overheads) will be given, but the fiber hours are calculated only with the on-source time (Jun/6/23)

* completeness vs. PPC (PPC is sorted by its total priority)
    * gray shades: regions with the total number of PPC exceeding the 5-night upper limit, should prevent your sample falling into this region
    * orange shades: regions covered by Grade A programs in last semester(s) / simulation
    * blue shades: regions covered by Grade B programs in last semester(s) / simulation
    * all the above information would be given by the observatory before CfP

* fiber usage fraction vs. PPC
    * average fraction: it is not recommended if this value is too low
    * fraction of PPC with the usage fraction less than 30%: it is not recommended if this value is too high

## Notes

* ~~overheads have not been included yet~~

* for some dense samples, 
    * it would take very long time to get the ouputs (>5 min)
    * ~~it may not achieve 100% completeness due to none fiber-assignment of netflow <span style="color:blue">(need to be fixed)</span>~~

* some functions have not been considered:
    * the optimized weighting parameters (conta,b&c) are taken from the simulation / last semester(s), they are not optimized for the input program in this tool
        * the output total observation time can be longer or shorter than that in the final observation
        * <span style="color:blue">need to check the discrepancy between the time estimated by the tool and the full PPP</span>
    * collision of fibers (trajectory) is not checked & corrected in this tool, as it will take a lot of time
        * for very dense sample (criteria? e.g., cluster candidates), should be careful 
        * <span style="color:blue">need to check its effect quantitatively</span>
    * calibrators
        * mock calibrators in the field
        * need to check the running time/outputs with & without calibrators
        
### Updates Jun/15/2023
* (Jun/6/23) overheads have been taken into account now, refer to [here](https://colab.research.google.com/drive/17a-hxaKZlGIAdLDKK0tMEHFT5AT-Ql7J?usp=sharing)
    * both the best and worst cases have been considered
    	* best: only one set of flat and arc would be taken for one night
        * worst: flat and arc would be taken for each fiber configuration
        
* (Jun/12/23) minor bugs fixed   
	* prevent blank pointings 

* (Jun/13/23) collision of fibers   
	* KDE part is also enabled to consider the collision of targets 
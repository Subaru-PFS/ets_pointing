# PPP online tool

This tool is designed for users preparing the PFS open-use proposals. It could help users to:

* estimate the total required hours / fiber hours;
* know the expected completeness of the user sample as a function of PPC (PFS pointing centers);
* evaluate the fiber usage fraction of each PPC.


### Executing program
* make a target list including "obj_id", "ra", "dec", "equinox", "priority", "exptime", "resolution" 
    * "ra" & "dec" should be given in unit of degree or hms,dms
    * "priority" is the inner priority defined by users, it should be gievn as an integer between 0 to 9, with 0 meaning the highest priority
    * "exptime" is the required observation time for each target, it should be given in unit of seconds
    * "resolution" can be 'L' (low-r) or 'M' (medium-r), default is 'L'
    * the format can be csv, ecsv or fits (astropy.Table can easily generate the file, refer to [here](https://github.com/Subaru-PFS/ets_target_database/blob/main/docs/input_target_list.md))
* upload the file 
* click button "start" 
	* please click button following the order Step 2 -> 3 -> 4
	* it can show PPCs determined by Step 2 if "Plot the results" is selected
* Results would be shown if "show results xxx" is selected
	* only results under the selected mode would be shown
    	* if the mode is not requested in the uploaded file, nothing will appear
        * if both the low- and medium-resolution modes are requested, please select "show results of the L&M-resolution mode" to see the summed up results
    * please modify the number of PPC by using the slider if needed, then the outputs would be changed accordingly

### outputs
* total time required 
	* the time would be given in unit of hours and fiber hours
    	* the fiber hours are calculated only with the on-source time 
    * both the total on-source time and required time (including overheads) will be given
    	* for overheads, refer to [here](https://colab.research.google.com/drive/17a-hxaKZlGIAdLDKK0tMEHFT5AT-Ql7J?usp=sharing)
    		* "best": only one set of flat and arc would be taken for one night
       		* "worst": flat and arc would be taken for each fiber configuration (PPC)          
* completeness vs. PPC (PPC is sorted by its total priority)
    * gray shades: regions with the total number of PPC exceeding the 5-night upper limit, should prevent your sample falling into this region
    * orange shades: regions covered by Grade A programs in last semester(s) / simulation
    * blue shades: regions covered by Grade B programs in last semester(s) / simulation
    * the above information is given by the observatory before CfP    
* fiber usage fraction vs. PPC (PPC is sorted by its total priority)
    * average fraction: it is not recommended if this value is too low
    * fraction of PPC with the usage fraction less than 30%: it is not recommended if this value is too high
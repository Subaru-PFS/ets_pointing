# PPP v2

This tool is designed to determine PFS pointing centers (PPCs) for the PFS open-use proposals. It determined PPCs to:

* maximize the average completion rate of proposals
* maximize the average fiber usage fraction

## Basic scheme

* step1: determine PPCs by KDE peaks
    * the weighting scheme

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

* read target list
    * method 1 - from local folder, e.g.,
      ```
      readsamp_con={'mode':'local', 'localPath':'PATH_to_file/YourFile.csv'}
      ```
    * method 2 - from database, e.g.,
      ```
      readsamp_con={'mode':'db', 'dialect':'postgresql','user':'obsproc','pwd':'obsproc','host':'pfsa-db01',\
              'port':'5433','dbname':'targetdb_e2e_test','sql_query':sql_t}
      ```
      ('sql_query' is the SQL query to the DB, e.g.,
      ```
      sql_t='''
      SELECT 
        T.ob_code, T.ra, T.dec, T.epoch, T.priority, T.effective_exptime, CASE WHEN T.is_medium_resolution = False THEN 'L' ELSE 'M' END AS resolution,      T.proposal_id, P.rank, P.grade
      FROM target as T
      JOIN proposal as P on T.proposal_id = P.proposal_id; 
      '''
      ```)

* set the total on-source time allocated to the PFS open-use programs
    * the time should be given in unit of seconds
    * the time of the low- and medium-resolution modes should be given separately
    * the input time should be comparable to the total required time of the input samples, e.g., if there are 10 input programs requiring 200 pointings to complete, the input time is recommended to be longer than ~150*900 sec to ensure plausible outputs.
 



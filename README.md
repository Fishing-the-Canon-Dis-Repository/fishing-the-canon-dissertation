This repositories contains Python scripts and datasets relevant to the master's dissertation "Fishing the Canon: An Attempt to Categorize Novels via Their Network of Relations", submitted for the University of Oxford's MSc of Digital Scholarship 2024-25.


Glossary of files: 

 - "2000NovelVectorizationInHathi_Public.py" contains the vectorization script used inside the HathiAnalytics Research Data Calsuple. 

 - "2000NovelAnalysis_Public.py" contains the script used for all the data analysis in "Fishing the Canon". 

 - "requirements-hathi-legacy.txt" should be used with "2000NovelVectorizationInHathi_Public.py" because the Python and package versions are meant to accommodate the configuration of the Research Data Capsule. 

 - "requirements-analysis.txt" is meant to be used with "2000NovelAnalysis_Public.py" and is intended to run on an up-to-date machine. 

 - "ModernNovelsTM.txt" is the list of node IDs filtered for fist publishing date >= 1700 and category = "longfiction" from the list "manual_title_subset.tsv" from Ted Underwood et al.'s NovelTM dataset. 

 - "node_centrality_metrics.csv" contains each node's centrality evaluation metrics. 

 - "weighted_EP6.csv" contains the results of the Scholkemper & Schaub's equitable partition optimization algorithm.

 - "top10_novels.csv" contains the JSTOR results of the ten most characteristic nodes of each category. 

 - "BookVectorData.csv" contains all the data pertaining tot he book vectorization conducted in the Research Data Capsule. 

# darkweb-threat-analysis

**Project Summary**

This project analyses dark-web related datasets to uncover malicious behaviour patterns and classify traffic using machine-learning techniques. The goal is to understand how malware, ransomware, and encrypted darknet communication differ from normal activity through measurable network and metadata features.

Using publicly available, anonymised cybersecurity datasets, the project performs:

Data preprocessing & label normalisation

Random Forest classification to separate Normal vs Malware

Visualization through confusion matrices, pie charts, heatmaps, and trend-line graphs

Feature-importance analysis to identify which attributes most strongly indicate malicious behaviour

Reproducible pipeline execution, allowing any user to re-run the full process on fresh infrastructure

Overall, the project demonstrates how machine-learning and traffic analytics can support threat intelligence, detect malicious activity, and provide deeper insight into evolving dark-web cybercrime trends.

**Dataset Overview**

This project uses publicly available cybersecurity datasets that contain dark-web related traffic, malware metadata, and encrypted communication patterns. The datasets help analyse how malicious and normal behaviour appear in network flows and metadata.

The datasets are used in the project for:

Classifying traffic as Normal or Malware using Random Forest

Identifying trends and behaviour patterns across timestamps

Extracting feature importance to understand what attributes drive malicious activity

Visualising activity using charts (confusion matrix, pie chart, trend line)

All datasets used here are research-safe, anonymised, and publicly accessible, and they allow the entire ML pipeline to be reproduced on any machine.
** dataset download link **

network_traffic_data.csv
https://www.kaggle.com/datasets/mohdzia356/network-traffic-data-for-intrusion-detection?resource=download

Binary -2DSCombined.csv
https://www.kaggle.com/datasets/haradityamvavasthi/bccc-darknet-2025?select=Binary+-2DSCombined.csv

MultiTotalDS.csv
https://www.kaggle.com/datasets/haradityamvavasthi/bccc-darknet-2025?select=Binary+-2DSCombined.csv

Final_Dataset_without_duplicate.csv
https://zenodo.org/records/13890887?utm_source=chatgpt.com

Bras_features.csv
https://springernature.figshare.com/articles/dataset/Tracffic_data_from_real_network_environment/28380347

BitcoinHeistData.csv
https://www.kaggle.com/datasets/salmalidame/bitcoinheistdata

# ClusChurn
A prototype implementation of ClusChurn based on PyTorch. 

1. preprocess.py:
Preprocess messaging, discover and lens activity data in the data folder to generate act_data.pkl and link_data.pkl
it can also draws daily activity and link plots
input: data downloaded from google bigquery
output: data.pkl: list of 2-d np array (day*act /15*10)

2. fit.py:
Feature engeering. Computes means and lags of activity data, fit by kmeans and draw the plots
input: data.pkl
output: params.pkl: 3-d np array (#user*#act*2)
churns.pkl: 1-d np array (#user)
labels.pkl: labels: 2-d np array (#act*#user), centers: 3-d np array (#act*k*2)
it also generates param, mixed and single plots of all activities

3. kmeans.py (helper function)
Kmeans with automatic selection of k 

4. evaluate.py (helper function)
Computation of f1, jc and nmi scores between two clustering results

5. cross.py
Visualize the correlations among activities
input: labels.pkl
output: cross.txt (values in the grid plots of correlations)

6. multiview.py - our main clustering pipeline in the paper
Every user is replaced by its nearest cluster center, show the most significant combinations. The number of clusters K is automatically chosen.
input: labels.pkl
output: multiview.txt: K outstanding clusters
soft.pkl: soft labels of users to clusters based on t-distribution (#users*K)

7. link.py
evaluate link smoothing intuitions, compute the portion of linked same-type users
input: linkset.txt, multiview.pkl
output: print out portion of linked same-type users

8. regress.py
baseline algorithms for performance comparison
input: act_data.pkl, churn.pkl
output: accuracy, precision and recall of churn prediction

9. model.py - our main prediction pipeline in the paper
parallel lstm model -- a simplified implementation with pooling.
input: act_data.pkl, churn.pkl
output: accuracy, precision and recall of churn prediction

10. core.py
find the core and evaluate its overlap with new users' friends
input: core_user, core_friend, core_degree (raw data from big query extraction)
output: print out percentage of friends in core, save the degree distribution figure

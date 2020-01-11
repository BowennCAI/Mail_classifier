import pandas as pd
import numpy as np

nn = pd.read_csv('knn_sample_submission.csv', index_col=0)
nn2 = pd.read_csv('nn_submission.csv', index_col=0)
rdf = pd.read_csv('rdf_submission.csv', index_col=0)
gb = pd.read_csv('GB_submission.csv', index_col=0)



data = np.hstack((nn, rdf, gb, nn2))
print(data)
result = []
i = 0
for a in data:
    counts = np.bincount(a)
    # 返回众数
    result.append(np.argmax(counts))

print(result)

pred_df = pd.DataFrame(result, columns=['label'])
pred_df.to_csv("compre_submission.csv", index=True, index_label='Id')

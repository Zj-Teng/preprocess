import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA


def main():
    exp_frame = pd.read_csv(
        'data/Benchmark Dataset/STRING Dataset/mHSC-L/TFs+1000/BL--ExpressionData.csv', index_col=0, header=0
    )

    transformer = FastICA(n_components=200)
    features = transformer.fit_transform(exp_frame)

    ret_frame = pd.DataFrame(features, index=list(exp_frame.index))
    print(ret_frame)
    ret_frame.to_csv('out/Benchmark Dataset/STRING Dataset/mHSC-L/TFs1000/ExpressionData.csv')

    return ret_frame


if __name__ == '__main__':
    main()

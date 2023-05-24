from sklearn.metrics import mean_squared_error,mean_absolute_error
import util_functions as uf
FACTOR = 8

def calc_errors(model, samples, labels, median=3, verbose=0):
    nlabels = model.predict(samples) 
    mse = 0
    mae = 0
    for i in range(labels.shape[0]):
        mse = mse + mean_squared_error(labels[i], uf.upsample_reg(samples[i],FACTOR))
        mae = mae + mean_absolute_error(labels[i], uf.upsample_reg(samples[i],FACTOR))
    mse = mse / labels.shape[0]
    mae = mae / labels.shape[0]
    if verbose==1:
        print('MSE_REG=',mse)
        print('MAE_REG=',mae)
    mse_reg = mse
    mse = 0
    mae = 0
    for i in range(labels.shape[0]):
        mse = mse + mean_squared_error(labels[i], nlabels[i])
        mae = mae + mean_absolute_error(labels[i], nlabels[i])
    mse = mse / labels.shape[0]
    mae = mae / labels.shape[0]
    if verbose==1:
        print('MSE_CNN=',mse)
        print('MAE_CNN=',mae)
    mse_cnn = mse
    mse = 0
    mae = 0
    mlabels = uf.median_f(nlabels,3)
    for i in range(labels.shape[0]):
        mse = mse + mean_squared_error(labels[i], mlabels[i])
        mae = mae + mean_absolute_error(labels[i], mlabels[i])
    mse = mse / labels.shape[0]
    mae = mae / labels.shape[0]
    mse_cnn_med = mse
    if verbose==1:
        print('MSE_CNN_MED=',mse)
        print('MAE_CNN_MED=',mae)
    return mse_cnn,mse_reg,mse_cnn_med
    
#calc_errors(model,dxs[64],dxl[64])
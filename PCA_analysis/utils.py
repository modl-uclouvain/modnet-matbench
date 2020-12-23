import pandas as pd
import numpy as np
from plotly import graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import pearsonr

class error_analysis:
    
    def __init__(self, name, key,n_feat=10000,multi=None):
        self.x_train = []
        self.x_test = []
        self.errors = []

        for f in range(5):
            self.df_train = pd.read_csv('../{}/folds/train_f{}.csv'.format(name,f+1),index_col=0)
            self.df_test = pd.read_csv('../{}/folds/test_f{}.csv'.format(name,f+1),index_col=0)
            x_train_f =  self.df_train.values
            if multi is not None:
                x_test_f =  self.df_test.drop(multi,axis=1).values
            else:
                x_test_f =  self.df_test.drop(key,axis=1).values
            errors_f =  self.df_test[key].values

            self.x_train.append(x_train_f[:,:n_feat])
            self.x_test.append(x_test_f[:,:n_feat])
            self.errors.append(errors_f)

        self.x = np.concatenate(self.x_test)

        self.x_train_sc = []
        self.x_test_sc = []
        self.scaler = StandardScaler()
        self.x_sc = self.scaler.fit_transform(self.x)
        for f in range(5):
            self.x_train_sc.append(self.scaler.fit_transform(self.x_train[f]))
            self.x_test_sc.append(self.scaler.fit_transform(self.x_test[f]))

        self.x_train_scmm = []
        self.x_test_scmm = []
        self.scaler = MinMaxScaler()
        self.x_scmm = self.scaler.fit_transform(self.x)
        for f in range(5):
            self.x_train_scmm.append(self.scaler.fit_transform(self.x_train[f]))
            self.x_test_scmm.append(self.scaler.fit_transform(self.x_test[f]))
        
        
    def run_pca(self, n_pc=10):
        self.pca = PCA(n_components=n_pc)
        self.pc = self.pca.fit_transform(self.x_sc)
        
        self.x_train_pc = []
        self.x_test_pc = []
        for f in range(5):
            self.x_train_pc.append(self.pca.transform(self.x_train_sc[f]))
            self.x_test_pc.append(self.pca.transform(self.x_test_sc[f]))
        
        for comp_n in range(3):
            print('-'*50)
            print('Component {}'.format(comp_n+1))
            print('-'*50)
            max_ids = np.abs(self.pca.components_[comp_n]).argsort()[::-1][:10]
            for f,a in zip(self.df_test.columns[max_ids],self.pca.components_[comp_n,max_ids]):
                print('{:.1f}% of {}'.format(a*100,f))
            print('\n')
            
        print('Explained variance')
        for i,v in enumerate(self.pca.explained_variance_ratio_):
            print('PC {}: {:.1f}%'.format(i+1,v*100))
        
    def plot_pca(self,pc1=1,pc2=2):
        fig, axs = plt.subplots(2,3,figsize=(25,15))

        for fold in range(5):
            ax = (axs.flat)[fold]
            ax.scatter(self.x_train_pc[fold][:,pc1-1],self.x_train_pc[fold][:,pc2-1],c='k',alpha=0.5)
            sc = ax.scatter(self.x_test_pc[fold][:,pc1-1],self.x_test_pc[fold][:,pc2-1],marker='X',c=np.log(1+np.abs(self.errors[fold])))
            fig.colorbar(sc, ax=ax)
            ax.set_xlabel('PC {}'.format(pc1))
            ax.set_ylabel('PC {}'.format(pc2))
            ax.set_title('fold {}'.format(fold+1))

            #MAE
            mae = np.mean(np.absolute(self.errors[fold]))
            ax.text(0.6,0.8,'MAE: {:.2f}'.format(mae),transform=ax.transAxes,fontsize=14)
        return fig, axs
        
        
    def plot_pca_3D(self,fold=0,classification=False):
        xd = np.concatenate([self.x_train_pc[fold],self.x_test_pc[fold]])
        if classification:
            colors = np.concatenate([np.ones(len(self.x_train_pc[fold]))*-1,np.log(1+np.abs(self.errors[fold]))])
        else:
            colors = np.concatenate([np.ones(len(self.x_train_pc[fold]))*-0.1,np.log(1+np.abs(self.errors[fold]))])
        symbols = ['circle']*len(self.x_train_pc[fold]) + ['x']*len(self.x_test_pc[fold])
        
        fig=go.Figure(data=go.Scatter3d(x=xd[:,0],y=xd[:,1],z=xd[:,2],
                                        mode='markers',marker=dict(size=4,color=colors,symbol=symbols,showscale=True)))
        fig.show()
        return fig
    
    def plot_pca_distance(self,n_neighbours = 5, n_pc = 10, xmax=None, ymax=None):

        fig, axs = plt.subplots(2,3,figsize=(25,15))
        all_closest = []
        for fold in range(6):
            if fold == 5:
                closest = np.concatenate(all_closest)
                errs = np.abs(np.concatenate(self.errors))
            else:
                train = self.x_test_pc[fold]
                test = self.x_test_pc[fold]
                dist = euclidean_distances(test[:,:n_pc],train[:,:n_pc])
                closest = np.sort(dist,axis=1)[:,:n_neighbours].mean(axis=1)
                all_closest.append(closest)
                errs = np.abs(self.errors[fold])


            ax = axs.flat[fold]
            ax.plot(closest,errs,'o',alpha=0.5)

            # 1st order fit
            pfit = np.polyval(np.polyfit(closest,errs,1),closest)
            ax.plot(closest,pfit,'--')

            # Pearson correlation
            corr = pearsonr(closest,errs)[0]
            ax.text(0.6,0.9,'corr: {:.2f}'.format(corr),transform=ax.transAxes,fontsize=14)

            #MAE
            mae = np.mean(errs)
            ax.text(0.6,0.8,'MAE: {:.2f}'.format(mae),transform=ax.transAxes,fontsize=14)

            ax.set_xlabel('PCA euclidean distance')
            ax.set_ylabel('Absolute error')
            if xmax is not None:
                ax.set_xlim(0,xmax)
            if ymax is not None:
                ax.set_ylim(0,ymax)
        return fig, axs
    
    def plot_feat_distance(self, n_neighbours = 5, n_feat = 50, scaling= 'mm', abs_v=True, xmax=None, ymax=None):
        
        #####
        fig, axs = plt.subplots(2,3,figsize=(25,15))

        if scaling == 'mm':
            xd_train = self.x_train_scmm
            xd_test = self.x_test_scmm
        else:
            xd_train = self.x_train_sc
            xd_test = self.x_test_sc

        all_closest = []
        for fold in range(6):
            if fold == 5:
                closest = np.concatenate(all_closest)
                if abs_v:
                    errs = np.abs(np.concatenate(self.errors))
                else:
                    errs = np.concatenate(self.errors)
            else:
                train = xd_test[fold]
                test = xd_test[fold]
                dist = euclidean_distances(test[:,:n_feat],train[:,:n_feat])
                closest = np.sort(dist,axis=1)[:,:n_neighbours].mean(axis=1)
                all_closest.append(closest)
                if abs_v:
                    errs = np.abs(self.errors[fold])
                else:
                    errs =self.errors[fold]


            ax = axs.flat[fold]
            ax.plot(closest,errs,'o',alpha=0.5)

            # 1st order fit
            pfit = np.polyval(np.polyfit(closest,errs,1),closest)
            ax.plot(closest,pfit,'--')

            # Pearson correlation
            corr = pearsonr(closest,errs)[0]
            ax.text(0.6,0.9,'corr: {:.2f}'.format(corr),transform=ax.transAxes,fontsize=14)

            #MAE
            mae = np.mean(errs)
            ax.text(0.6,0.8,'MAE: {:.2f}'.format(mae),transform=ax.transAxes,fontsize=14)

            ax.set_xlabel('PCA euclidean distance')
            ax.set_ylabel('Absolute error')
            if xmax is not None:
                ax.set_xlim(0,xmax)
            if ymax is not None:
                ax.set_ylim(0,ymax)
        return fig, axs
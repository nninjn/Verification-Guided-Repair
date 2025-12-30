import numpy as np
from network import FNN_gen
import torch


# property 2
# "bounds": "[(0.6,0.6799), (-0.5,0.5), (-0.5,0.5), (0.45,0.5), (-0.5,-0.45)]",
# please note that we use a strengthened version of property 2 where requir the min score is 0
def __generate_x():
    x_0 = np.random.uniform(low=0.6, high=0.6799, size=(BATCH_SIZE, 1))
    x_1 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_2 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_3 = np.random.uniform(low=0.45, high=0.5, size=(BATCH_SIZE, 1))
    x_4 = np.random.uniform(low=-0.5, high=-0.45, size=(BATCH_SIZE, 1))
    x = np.array([x_0, x_1, x_2, x_3, x_4])
    return np.transpose(x.reshape(5,32))

def property_satisfied(pre_y):
    #pre_y = np.argmin(y, axis=1)
    if pre_y == 0:
        return True
    return False

def gen_data_set(model, data_path, ty):
    """generate drawndown and counter example data set"""
    # property 2
    n_dd = 0
    n_cex = 0
    dd = torch.empty([DRAWNDOWN_SIZE, 5])
    cex = torch.empty([COUNTEREG_SIZE, 5])
    print(dd.shape, cex.shape)
    dd_saved = 0
    cex_saved = 0

    while True:
        x = __generate_x()
        x = torch.Tensor(x).cuda()
        print(x.shape, x.device)
        y = model(x.reshape(BATCH_SIZE,5))
        pre_y =  torch.argmin(y, axis=1)

        for i in range (0, BATCH_SIZE):
            if (property_satisfied(pre_y[i])):
                if n_dd < DRAWNDOWN_SIZE:
                    dd[n_dd] = x[i].detach().clone()
                    n_dd = n_dd + 1
                    print('n_dd:{}'.format(n_dd))
            else:
                if n_cex < COUNTEREG_SIZE:
                    cex[n_cex] = x[i].detach().clone()
                    n_cex = n_cex + 1
                    print('n_cex:{}'.format(n_cex))

        if n_dd >= DRAWNDOWN_SIZE and dd_saved == 0:
            dd_saved = 1
            # dd = torch.Tensor(dd)
            print('dd ', dd.shape)
            torch.save(dd, data_path + '/data/drawdown' + ty + '.pt')


        if n_cex >= COUNTEREG_SIZE and cex_saved == 0:
            cex_saved = 1
            # cex = torch.Tensor(cex)
            print('cex ', cex.shape)
            torch.save(cex, data_path + '/data/counterexample' + ty + '.pt')
            print(data_path)

        if dd_saved == 1 and cex_saved == 1:
            break
    return n_dd, n_cex


def gen_data_set_Fidelity(model, data_path, ty):
    """generate drawndown and counter example data set"""
    # property 2
    n_dd = 0
    Fid = torch.empty([Fidelity_SIZE, 5])
    print(Fid.shape)
    dd_saved = 0

    while True:
        x = __generate_x()
        x = torch.Tensor(x).cuda()
        print(x.shape, x.device)
        y = model(x.reshape(BATCH_SIZE,5))
        pre_y =  torch.argmin(y, axis=1)

        for i in range (0, BATCH_SIZE):
            if n_dd < Fidelity_SIZE:
                Fid[n_dd] = x[i].detach().clone()
                n_dd = n_dd + 1
                print('n_dd:{}'.format(n_dd))

        if n_dd >= Fidelity_SIZE and dd_saved == 0:
            dd_saved = 1
            # dd = torch.Tensor(dd)
            print('Fidelity ', Fid.shape)
            print('Fidelity ', Fid)
            # torch.save(Fid, data_path + '/data/Fidelity_test.pt')
        if dd_saved == 1:
            break
    return n_dd

BATCH_SIZE = 32
DRAWNDOWN_SIZE = 32 * 313
COUNTEREG_SIZE = 32 * 313
Fidelity_SIZE = 5000
Fidelity_SIZE = 10
model = FNN_gen().cuda()
model.load_state_dict(torch.load('n19/model/n19.pth'))
pp = './n19'
gen_data_set_Fidelity(model, pp, '')
# for i in range(1, 10):
#     pp = './n5' + str(i)
#     ppp = 'n5' + str(i) + '/model/n5' + str(i) + '.pth'
#     model.load_state_dict(torch.load(ppp))
#     # gen_data_set(model, 'n21/data')
#     gen_data_set(model, pp, '')
#     gen_data_set(model, pp, '_test')
#     # gen_data_set(model, './n43', '')
#     # gen_data_set(model, './n43', '_test')

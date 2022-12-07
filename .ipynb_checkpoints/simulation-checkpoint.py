import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 16})


class SimClass:
    def __init__(self, param):
        ## Battery Params
        self.param = param
        self.Qbatt = 1.2  # Ah
        self.dt    = 1    # sec

        ## Temperature Params
        self.Rc = 1.94
        self.Cc = 62.7
        self.Ru = 3.19
        self.Cs = 4.5
        

    def setCurrent(self, current):
        # if isinstance(current, pd.DataFrame):
        #     self.current = current['Current (A)'].values
        # else:
        #     self.current = current
        self.current = current


    def setInitSoC(self, initSoC):
        self.init_SOC = initSoC


    def setInitTemp(self, Tc = 25, Ts = 25, Tf = 25):
        self.init_Tc = Tc
        self.init_Ts = Ts
        self.init_Tf = Tf


    def setOCV(self, data, order = 7, figOn = 'ON'):
        x = np.linspace(1, 0, len(data))
        coeff = np.polyfit(x, data['Voltage (V)'], order)
        SOC2OCV = np.poly1d(coeff)

        if figOn == 'ON':
            plt.plot(x, data['Voltage (V)'], 'b')
            plt.plot(x, SOC2OCV(x), 'r--')
            plt.legend(['True', 'Fitted'])
            plt.title('OCV vs SOC')
            plt.ylabel('Voltage (V)')
            plt.xlabel('SOC')
        
        self.SOC2OCV = SOC2OCV


    def getOCV(self, SOC):
        return self.SOC2OCV(SOC)


    def getParam(self, SOC):
        # print('SOC', SOC)
        # print('SOC', type(SOC))
        if 0.9 >= SOC > 0.8:
            # temp = self.param[self.param['SOC'] == 0.9]
            temp = self.param.iloc[0]
        elif 0.8 >= SOC > 0.7:
            # temp = self.param[self.param['SOC'] == 0.8]
            temp = self.param.iloc[1]
        elif 0.7 >= SOC > 0.6:
            # temp = self.param[self.param['SOC'] == 0.7]
            temp = self.param.iloc[2]
        elif 0.6 >= SOC > 0.5:
            # temp = self.param[self.param['SOC'] == 0.6]
            temp = self.param.iloc[3]
        elif 0.5 >= SOC > 0.4:
            # temp = self.param[self.param['SOC'] == 0.5]
            temp = self.param.iloc[4]
        elif 0.4 >= SOC > 0.3:
            # temp = self.param[self.param['SOC'] == 0.4]
            temp = self.param.iloc[5]
        elif 0.3 >= SOC > 0.2:
            # temp = self.param[self.param['SOC'] == 0.3]
            temp = self.param.iloc[6]
        elif 0.2 >= SOC > 0.1:
            # temp = self.param[self.param['SOC'] == 0.2]
            temp = self.param.iloc[7]
        elif 0.1 >= SOC > 0:
            # temp = self.param[self.param['SOC'] == 0.1]
            temp = self.param.iloc[8]

        # print()
        # print('temp: ', temp)
        # print('R0: ', temp['R0'])
        # print('R1: ', temp['R1'])
        # print('R2: ', temp['R2'])
        # print('C1: ', temp['C1'])
        # print('C2: ', temp['C2'])

        # return temp['R0'].values[0], temp['R1'].values[0], temp['R2'].values[0], temp['C1'].values[0], temp['C2'].values[0]
        return temp['R0'], temp['R1'], temp['R2'], temp['C1'], temp['C2']



    def plotSimResult(self, sim_df):
        fig, axs = plt.subplots(3, 3, figsize=(30, 20))
        axs[0, 0].plot(sim_df['time'], sim_df['I'], 'b')
        axs[0, 0].set_title('Current')
        axs[0, 0].set_ylabel('I [A]')
        axs[0, 0].set_xlabel('t [s]')

        axs[0, 1].plot(sim_df['time'], sim_df['V'], 'b')
        axs[0, 1].set_title('Voltage')
        axs[0, 1].set_ylabel('V [V]')
        axs[0, 1].set_xlabel('t [s]')

        axs[0, 2].plot(sim_df['time'], sim_df['SOC'], 'b')
        axs[0, 2].set_title('SOC')
        axs[0, 2].set_ylabel('SOC [-]')
        axs[0, 2].set_xlabel('t [s]')

        axs[1, 0].plot(sim_df['time'], sim_df['Tc'], 'b')
        axs[1, 0].set_title('Core Temperature')
        axs[1, 0].set_ylabel('Tc [C]')
        axs[1, 0].set_xlabel('t [s]')

        axs[1, 1].plot(sim_df['time'], sim_df['Ts'], 'b')
        axs[1, 1].set_title('Surface Temperature')
        axs[1, 1].set_ylabel('Ts [C]')
        axs[1, 1].set_xlabel('t [s]')

        axs[2, 0].plot(sim_df['time'], sim_df['Vc1'], 'b')
        axs[2, 0].set_title('Vc1')
        axs[2, 0].set_ylabel('V [V]')
        axs[2, 0].set_xlabel('t [s]')

        axs[2, 1].plot(sim_df['time'], sim_df['Vc2'], 'b')
        axs[2, 1].set_title('Vc2')
        axs[2, 1].set_ylabel('V [V]')
        axs[2, 1].set_xlabel('t [s]')

        plt.tight_layout()



    def runSimulation(self):
        self.batt_sim         = pd.DataFrame()
        self.batt_sim['time'] = np.arange(0, len(self.current))
        self.batt_sim['I']    = self.current
        self.batt_sim['Tc']   = self.init_Tc
        self.batt_sim['Ts']   = self.init_Ts
        self.batt_sim['SOC']  = self.init_SOC
        self.batt_sim['V']    = self.SOC2OCV(self.init_SOC)
        self.batt_sim['Vc1']  = 0
        self.batt_sim['Vc2']  = 0

        SOC  = self.init_SOC
        R0, R1, R2, C1, C2 = self.getParam(SOC) 
        # print()
        # print('R0, R1, R2, C1, C2', R0, R1, R2, C1, C2)
        Vc1  = 0
        Vc2  = 0
        Tc  = self.init_Tc
        Ts  = self.init_Ts

        # print('SOC: ', SOC)
        # print('Vc1: ', Vc1)
        # print('Vc2: ', Vc2)
        # print('Tc: ', Tc)
        # print('Ts: ', Ts)

        for k in range(len(self.current)-1):
            if k%1000 == 0:
                print('k: ', k)

            ## output update
            OCV = self.getOCV(SOC)
            I   = self.batt_sim['I'].iloc[k]
            V   = OCV - Vc1 - Vc2 - I*R0
            # print('OCV: ', OCV)
            # print('I: ', I)
            # print('V: ', V)
            
            ## state update
            SOC = SOC + self.dt * ( -I / (self.Qbatt * 3600) )
            Vc1 = Vc1 + self.dt * ( -Vc1/(R1*C1) + I/C1 )
            Vc2 = Vc2 + self.dt * ( -Vc2/(R2*C2) + I/C2 )
            # Q   = abs( I * ( OCV - V ) )
            Q   = 100 * abs( I * ( OCV - V ) )
            self.batt_sim['Tc'].iloc[k+1]  = Tc + self.dt * ( (Ts-Tc)/(self.Rc*self.Cc) + Q/self.Cc )
            self.batt_sim['Ts'].iloc[k+1]  = Ts + self.dt * ( (self.init_Tf-Ts)/(self.Ru*self.Cs) - (Ts-Tc)/(self.Rc*self.Cs) )

            Tc = self.batt_sim['Tc'].iloc[k+1]
            Ts = self.batt_sim['Ts'].iloc[k+1]
            # print('SOC: ', SOC)
            # print('Vc1: ', Vc1)
            # print('Vc2: ', Vc2)
            # print('Q: ', Q)
            # print('Tc: ', Tc)
            # print('Ts: ', Ts)


            ## batt params update
            R0, R1, R2, C1, C2 = self.getParam(SOC) 
            # print('R0, R1, R2, C1, C2', R0, R1, R2, C1, C2)

            ## store data
            self.batt_sim['V'].iloc[k]     = V
            self.batt_sim['SOC'].iloc[k+1] = SOC
            self.batt_sim['Vc1'].iloc[k+1] = Vc1
            self.batt_sim['Vc2'].iloc[k+1] = Vc2

            
        ## get last output update
        self.batt_sim['V'].iloc[-1] = self.getOCV(self.batt_sim['SOC'].iloc[-1]) - self.batt_sim['Vc1'].iloc[-1] - self.batt_sim['Vc2'].iloc[-1] - self.batt_sim['I'].iloc[-1] * R0

        self.plotSimResult(self.batt_sim)

        return self.batt_sim


























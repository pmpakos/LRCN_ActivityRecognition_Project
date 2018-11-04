from prettytable import PrettyTable
import numpy as np
import operator

def bold(str):
    return '\033[1m' + str +'\033[0m'

def ModelsComparison():
    f1 = open('results/single_frame_results/single_frame_results.txt','r')
    f2 = open('results/video_sequence_results/video_sequnce_results.txt','r')
    l1 = f1.readlines()
    l2 = f2.readlines()
    f1.close()
    f2.close()

    # Single Frame
    # Single Input Type
    caffenet_single_rgb = l1[2].split()[7]
    inception_rgb = l1[5].split()[7]
    caffenet_single_flow = l1[8].split()[7]

    # Weighted Average
    rgb_flow_067_033 = l1[13].split()[7]
    rgb_flow_05_05 = l1[14].split()[7]
    rgb_flow_033_067 = l1[15].split()[7]

    inception_flow_067_033 = l1[18].split()[7]
    inception_flow_05_05 = bold(l1[19].split()[7]) # MAX
    inception_flow_033_067 = l1[20].split()[7]

    # LRCN
    # Single Input Type
    lstm_rgb = l2[2].split()[7]
    lstm_flow512 = l2[5].split()[7]
    lstm_flow1024 = l2[8].split()[7]

    # Weighted Average
    rgb_flow512_067_033 = l2[13].split()[7]
    rgb_flow512_05_05 = l2[14].split()[7]
    rgb_flow512_033_067 = bold(l2[15].split()[7]) # MAX

    rgb_flow1024_067_033 = l2[18].split()[7]
    rgb_flow1024_05_05 = l2[19].split()[7]
    rgb_flow1024_033_067 = l2[20].split()[7]

    t = PrettyTable(['',bold('Single Input Type'), bold('Weighted Average')])
    t.add_row([bold('Model'),bold('RGB  ')+' | '+bold('  Flow'), bold('(2/3 1/3)') + ' | ' + bold('(1/2 1/2)') + ' | ' + bold('(1/3 2/3)')])
    t.add_row(['---------------------------------','-----------------', '----------------------------------'])
    t.add_row([bold('Single Frame RGB + Flow'),caffenet_single_rgb+' |  '+caffenet_single_flow, rgb_flow_067_033+'  |  '+rgb_flow_05_05+'   |   '+rgb_flow_033_067])
    t.add_row([bold('Inception RGB + Single Frame Flow'),inception_rgb+' |  '+caffenet_single_flow, inception_flow_067_033+'  |  '+inception_flow_05_05+'   |   '+inception_flow_033_067])
    t.add_row(['---------------------------------','-----------------', '----------------------------------'])
    t.add_row([bold('LRCN (RGB 512 + Flow 512)'),lstm_rgb+' |  '+lstm_flow512, rgb_flow512_067_033+'  |  '+rgb_flow512_05_05+'   |   '+rgb_flow512_033_067])
    t.add_row([bold('LRCN (RGB 512 + Flow 1024)'),lstm_rgb+' |  '+lstm_flow1024, rgb_flow1024_067_033+'  |  '+rgb_flow1024_05_05+'   |   '+rgb_flow1024_033_067])
    print(t)

def PerCategoryRatioComparison(modelname):
    if(modelname=='Single Frame RGB+Flow'):
        curr = 'single_frame_results/combo0/single_frame_stats'
    elif(modelname=='Inception RGB + Single Frame Flow'):
        curr = 'single_frame_results/combo1/single_frame_stats'
    elif(modelname=='LRCN (RGB 512 + Flow 512)'):
        curr = 'video_sequence_results/combo0/video_sequence_stats'
    elif(modelname=='LRCN (RGB 512 + Flow 1024)'):
        curr = 'video_sequence_results/combo1/video_sequence_stats'
    else:
        print('Wrong modelname given')
        return
    f1 = open('results/'+curr+'_0.0_1.0.txt','r')
    f2 = open('results/'+curr+'_0.67_0.33.txt','r')
    f3 = open('results/'+curr+'_0.5_0.5.txt','r')
    f4 = open('results/'+curr+'_0.33_0.67.txt','r')
    f5 = open('results/'+curr+'_1.0_0.0.txt','r')

    l1 = f1.readlines()
    l2 = f2.readlines()
    l3 = f3.readlines()
    l4 = f4.readlines()
    l5 = f5.readlines()

    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()

    ratio = ['0 - 1','2/3 - 1/3','1/2 - 1/2', '1/3 - 2/3','1 - 0']
    print('Per-Category Classification for different RGB-Flow ratios for',bold(modelname),'model')
    t = PrettyTable([bold('Category'),bold(ratio[0]),bold(ratio[1]),bold(ratio[2]),bold(ratio[3]),bold(ratio[4]),bold('Total')])

    # correct = np.zeros(5,dtype=np.int)
    for line1,line2,line3,line4,line5 in zip(l1,l2,l3,l4,l5):
        label = line1.split()[-1]
        total = line1.split()[4]
        c=[]
        c.append(int(line1.split()[2]))
        c.append(int(line2.split()[2]))
        c.append(int(line3.split()[2]))
        c.append(int(line4.split()[2]))
        c.append(int(line5.split()[2]))
        # ind = np.argmax(c)
        # c[ind] = bold(str(c[ind]))
        # correct[ind]+=1
        t.add_row([bold(label),c[0],c[1],c[2],c[3],c[4],bold(total)])
    # ind2 = np.argmax(correct)
    # print('Best Ratio for this model is',bold(ratio[ind2]))
    print(t)    

def Single_LCRN_Comparison():
    f1 = open('results/single_frame_results/combo0/single_frame_stats_0.33_0.67.txt','r')
    f2 = open('results/video_sequence_results/combo0/video_sequence_stats_0.33_0.67.txt','r')
    l1 = f1.readlines()
    l2 = f2.readlines()

    f1.close()
    f2.close()

    print('Per Category Improvement Comparison between Single Frame (1/3 RGB)-(2/3 Flow) Model and its equivalent LRCN')
    t = PrettyTable([bold('Category'),bold('Single Frame'),bold('LRCN'),bold('Improvement')])

    for line1,line2 in zip(l1,l2):
        label = line1.split()[-1]
        total = line1.split()[4]
        c=[]
        c.append(int(line1.split()[2]))
        c.append(int(line2.split()[2]))
        c.append(int(c[1]-c[0]))
        t.add_row([bold(label),c[0],c[1],c[2]]) 
    print (t.get_string(reversesort=True,sort_key=operator.itemgetter(4), sortby=bold('Improvement')))


def LCRN_RGB_Flow_Comparison():
    f1 = open('results/video_sequence_results/combo0/video_sequence_stats_1.0_0.0.txt','r')
    f2 = open('results/video_sequence_results/combo0/video_sequence_stats_0.0_1.0.txt','r')
    l1 = f1.readlines()
    l2 = f2.readlines()

    f1.close()
    f2.close()

    print('Per Category Improvement Comparison between RGB and Flow LRCN Model')
    print('Difference > 0 => Flow model better on these actions')
    print('Difference < 0 => RGB model better on these actions')
    t = PrettyTable([bold('Category'),bold('RGB'),bold('Flow'),bold('Difference')])

    for line1,line2 in zip(l1,l2):
        label = line1.split()[-1]
        total = line1.split()[4]
        c=[]
        c.append(int(line1.split()[2]))
        c.append(int(line2.split()[2]))
        c.append(int(c[1]-c[0]))
        t.add_row([bold(label),c[0],c[1],c[2]]) 
    print (t.get_string(reversesort=True,sort_key=operator.itemgetter(4), sortby=bold('Difference')))

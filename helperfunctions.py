import datetime
import matplotlib.pyplot as plt
try:
	import winsound
except:
	pass

def datestring():
    return datetime.datetime.today().strftime('%Y-%m-%d %H;%M;%S')

def savefigure(savename):
    try:
        plt.savefig(savename + '.svg', dpi = 600, bbox_inches='tight', transparent=True)
    except:
        print('Could not save svg')
    try:
        plt.savefig(savename + '.pdf', dpi = 600, bbox_inches='tight', transparent=True)
           # transparent true source: https://jonathansoma.com/lede/data-studio/matplotlib/exporting-from-matplotlib-to-open-in-adobe-illustrator/
    except:
        print('Could not save pdf')
    plt.savefig(savename + '.png', dpi = 600, bbox_inches='tight', transparent=True)
    print("Saved:\n", savename + '.png')

def beep():
    try:
        winsound.PlaySound(r'C:\Windows\Media\Speech Disambiguation.wav', flags = winsound.SND_ASYNC)
        return
    except:
        pass
    try:
        winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
        return
    except:
        pass
    try:
        winsound.Beep(450,150)
        return
    except:
        pass

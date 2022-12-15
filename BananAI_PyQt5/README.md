To Run Program with Splash Screen effect: Need SplashScreen File (File Too Big for Github)


To Run Without Splash Screen Comment out:

Lines 112-119:

'mySplash = MovieSplashScreen("./banana_logo_splash.gif")
    mySplash.show()

    def showWindow():
        mySplash.close()
        window.show()

    QtCore.QTimer.singleShot(6000, showWindow)'
    
Then Uncomment line 121:
'# window.show()'



For Different Models, change modelname(number)
Currently model 0 and 1

from gym.envs.registration import register

register(
    id='Eplus-demo-v1',
    entry_point='energym.envs:EplusDiscrete',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV.idf',
        'weather_file': 'USA_PA_Pittsburgh-Allegheny.County.AP.725205_TMY3.epw' 
    }
)

register(
    id='Eplus-discrete-hot-v1',
    entry_point='energym.envs:EplusDiscrete',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw' 
    }
)

register(
    id='Eplus-discrete-mixed-v1',
    entry_point='energym.envs:EplusDiscrete',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw' 
    }
)

register(
    id='Eplus-discrete-cool-v1',
    entry_point='energym.envs:EplusDiscrete',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw' 
    }
)

register(
    id='Eplus-continuous-cool-v1',
    entry_point='energym.envs:EplusContinuous',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_WA.idf',
        'weather_file': 'USA_WA_Port.Angeles-William.R.Fairchild.Intl.AP.727885_TMY3.epw' 
    }
)

register(
    id='Eplus-continuous-mixed-v1',
    entry_point='energym.envs:EplusContinuous',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_NY.idf',
        'weather_file': 'USA_NY_New.York-John.F.Kennedy.Intl.AP.744860_TMY3.epw' 
    }
)

register(
    id='Eplus-continuous-hot-v1',
    entry_point='energym.envs:EplusContinuous',
    kwargs={
        'idf_file': '5ZoneAutoDXVAV_AZ.idf',
        'weather_file': 'USA_AZ_Tucson-Davis-Monthan.AFB.722745_TMY3.epw' 
    }
)
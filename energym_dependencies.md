graph TD
    %% Main package structure
    setup[setup.py] --> energym[energym/__init__.py]
    requirements[requirements.txt] --> setup
    
    %% Core modules
    energym --> utils[energym/utils/__init__.py]
    energym --> envs[energym/envs/__init__.py]
    energym --> simulators[energym/simulators/__init__.py]
    
    %% Utils submodules
    utils --> common[energym/utils/common.py]
    utils --> controllers[energym/utils/controllers.py]
    utils --> rewards[energym/utils/rewards.py]
    utils --> wrappers[energym/utils/wrappers.py]
    
    %% Environment modules
    envs --> eplus_env[energym/envs/eplus_env.py]
    envs --> env_models[energym/envs/env_models.py]
    
    %% Simulator modules
    simulators --> eplus[energym/simulators/eplus.py]
    
    %% Dependencies between modules
    eplus_env --> eplus
    eplus_env --> common
    eplus_env --> rewards
    
    controllers --> common
    
    %% External dependencies
    docker[Dockerfile] -.-> setup
    docker -.-> requirements
    
    %% Documentation
    docs[doc/source/index.rst] -.-> energym
    api_ref[doc/source/pages/API-reference.rst] -.-> docs
    api_ref -.-> utils
    api_ref -.-> envs
    api_ref -.-> simulators
    
    %% Templates for documentation
    module_template[doc/source/_templates/custom-module-template.rst] -.-> api_ref
    class_template[doc/source/_templates/custom-class-template.rst] -.-> module_template
    
    %% User scripts
    notebooks[DQN with and without DA.ipynb] --> energym
    notebooks --> eplus_env
    notebooks --> controllers
    
    %% Testing
    tests[tests/] --> energym
    travis[.travis.yml] -.-> tests
    
    %% Legend
    classDef core fill:#f9f,stroke:#333,stroke-width:2px,color:#000;
    classDef utils fill:#bbf,stroke:#333,stroke-width:1px,color:#000;
    classDef envs fill:#bfb,stroke:#333,stroke-width:1px,color:#000;
    classDef simulators fill:#fbb,stroke:#333,stroke-width:1px,color:#000;
    classDef docs fill:#ffd,stroke:#333,stroke-width:1px,color:#000;
    classDef external fill:#ddd,stroke:#333,stroke-width:1px,color:#000;
    
    class energym,setup,requirements core;
    class utils,common,controllers,rewards,wrappers utils;
    class envs,eplus_env,env_models envs;
    class simulators,eplus simulators;
    class docs,api_ref,module_template,class_template docs;
    class docker,travis
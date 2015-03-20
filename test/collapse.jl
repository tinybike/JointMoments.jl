weighted_collapse = [ 20.5482
                      16.8602
                      10.0365 ]

weighted_collapse_cols = [ -2044.85  
                             133.236 
                             434.136 
                            7583.68  
                           -2541.21  
                             -14.7758
                           -1584.42  
                             354.702 
                            -245.613 
                             793.843 
                             972.923 
                            1104.47  
                            2093.7   
                            -728.741 
                            -550.393 
                            -674.196 
                            1772.04  
                            1253.68  
                            3944.08  
                            5514.91  
                            -646.185 
                              99.3528
                            1516.22  
                            1230.86  ]

@test tolerance(weighted_collapse, collapse(data, w, 4; standardize=true, axis=1))

@test tolerance(weighted_collapse_cols, collapse(data', w, 4; standardize=true, axis=2))

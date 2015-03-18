expected_coskew = [ 0.147575   0.154872   0.126323   0.154872   0.414994  -0.033756   0.126323  -0.033756   0.013083
                    0.154872   0.414994  -0.033756   0.414994   0.861588  -0.301254  -0.033756  -0.301254  -0.287069
                    0.126323  -0.033756   0.013083  -0.033756  -0.301254  -0.287069   0.013083  -0.287069   1.017824]

expected_cokurt = [ 2.126762   1.118844   0.474779   1.118844   1.122935   0.187329   0.474779   0.187329   1.155241   1.118844   1.122935   0.187329   1.122935   1.404624  -0.026636   0.187329  -0.026636   0.276558   0.474779   0.187329  1.155241   0.187329  -0.026636   0.276558   1.155241   0.276558   0.178076
                    1.118844   1.122935   0.187329   1.122935   1.404624  -0.026636   0.187329  -0.026636   0.276558   1.122935   1.404624  -0.026636   1.404624   3.102880  -0.517198  -0.026636  -0.517198   0.779222   0.187329  -0.026636  0.276558  -0.026636  -0.517198   0.779222   0.276558   0.779222   0.218735
                    0.474779   0.187329   1.155241   0.187329  -0.026636   0.276558   1.155241   0.276558   0.178076   0.187329  -0.026636   0.276558  -0.026636  -0.517198   0.779222   0.276558   0.779222   0.218735   1.155241   0.276558  0.178076   0.276558   0.779222   0.218735   0.178076   0.218735   5.989492]

expected_coskew_tensor = reshape(expected_coskew, 3, 3, 3)
expected_cokurt_tensor = reshape(expected_cokurt, 3, 3, 3, 3)

@test_approx_eq_eps _cov(data, bias=1) cov(data, corrected=true) ε
@test_approx_eq_eps _cov(data, bias=0) cov(data, corrected=false) ε

ltm = _cov(data, bias=0, dense=false)
@test_approx_eq_eps(ltm + ltm' - diagm(diag(ltm)), _cov(data, bias=0, dense=true), ε)

@test_approx_eq_eps coskew(data, flatten=true) expected_coskew ε
@test_approx_eq_eps coskew(data) expected_coskew_tensor ε
@test_approx_eq coskew(data, flatten=true)[1,4] coskew(data, flatten=true, dense=false)[1,4]
@test_approx_eq coskew(data, flatten=true, dense=false)[1,4] coskew(data, dense=false)[2,1,1]
@test_approx_eq coskew(data, dense=false)[2,1,1] coskew(data)[2,1,1]

@test_approx_eq_eps cokurt(data, flatten=true) expected_cokurt ε
@test_approx_eq_eps cokurt(data) expected_cokurt_tensor ε
@test_approx_eq cokurt(data, flatten=true)[1,10] cokurt(data, flatten=true, dense=false)[1,10]
@test_approx_eq cokurt(data, flatten=true, dense=false)[1,10] cokurt(data, dense=false)[2,1,1,1]
@test_approx_eq cokurt(data, dense=false)[2,1,1,1] cokurt(data)[2,1,1,1]

for d in (data, bigdata)
    covmat = _cov(d; bias=0)
    summed = sum(covmat, 2)[:]
    @test all(summed - coalesce(d, 2; bias=0) .< ε)

    tensor = coskew(d; standardize=true, bias=0)
    summed = sum(sum(tensor, 3), 2)[:]
    @test all(summed - coalesce(d, 3; standardize=true, bias=0) .< ε)

    tensor = cokurt(d; standardize=true, bias=0)
    summed = sum(sum(sum(tensor, 4), 3), 2)[:]
    @test all(summed - coalesce(d, 4; standardize=true, bias=0) .< ε)
end

weighted_center =  [ 0.0622965     0.0422869    0.133849   
                    -0.014896     -0.0147219    0.0260687  
                    -0.0741197    -0.0152463    0.0388971  
                    -0.0289413    -0.0152392   -0.0235819  
                     0.144315      0.053791     0.0291951  
                    -0.0154791    -0.0558604    0.0799623  
                     0.0722315     0.15276     -0.0250683  
                    -0.000462568   0.00446689  -0.0045595  
                    -0.0303173    -0.00982378   0.0535941  
                    -0.0085624    -0.0077157    0.0057796  
                     0.0135006    -0.0547161    0.00207633 
                     0.00335314   -0.0119657   -0.000843905
                    -0.043894     -0.00767854  -0.0780369  
                     0.0256303     0.0668316   -0.0349972  
                    -0.00643617    0.0435989   -0.0158489  
                     0.023722     -0.0208573    0.0474277  
                     0.00192959   -0.0182444   -0.00546336 
                     0.0341471    -0.0550279   -0.112117   
                    -0.00527763   -0.0137732   -0.0144545  
                    -0.119425     -0.0866519   -0.0435022  
                    -0.0106706     0.047902    -0.0131017  
                     0.0426258    -0.0430047    0.014838   
                    -0.0503171     0.0137864   -0.0511462  
                    -0.0149529     0.00510296  -0.00896664 ]

weighted_coalesce = [ 0.000210328
                      0.000163799
                      0.000129309 ]

@test all(center(data, w; standardize=true, bias=0)[1] - weighted_center .< ε)
@test all(coalesce(data, w, 4; standardize=true, bias=0)[1] - weighted_coalesce .< ε)

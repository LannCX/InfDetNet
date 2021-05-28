close all

test_file_list = {'res_r-c3d.txt', 'res_bsn.txt', 'res_bmn.txt', ...
    'res_sel_cross_atten.txt', 'sel_atten_joint_flow.txt', 'sel_ft_i3d.txt'};

for k = 1:length(test_file_list)
    file_name = test_file_list{k};
    ap = 0;
    for n=linspace(0.5,0.95,10)
        [~, ~, map] = TH14evalDet(file_name, 'annotation', 'test', 0.5);
        ap = ap+map;
    end
    fprintf('\n mAP across 0.5-0.95: %f',ap/10);
end
print('\n')
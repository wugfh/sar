clear; clc;

lambda = 0.031; % 载波波长

EarthMass = 6e24; % 地球质量(kg)
EarthRadius = 6.37e6; % 地球半径6371km
Gravitational = 6.67e-11; % 万有引力常量
incident_min = deg2rad(20);
incident_max = deg2rad(55); % 最大入射角
H = 1e3 * linspace(500, 700, 500);

gamma_min = asin(EarthRadius * sin(incident_min) ./ (H + EarthRadius)); % 下视角
beta_min = incident_min - gamma_min;

gamma_max = asin(EarthRadius * sin(incident_max) ./ (H + EarthRadius));
beta_max = incident_max - gamma_max;

% 一共测绘6条
Wg = EarthRadius * (beta_max - beta_min)/6;

figure("name","relation between H and Wg")
plot(H, Wg);
title("relation between H and Wg");

% 单条测绘宽度要>100km，
H = 580e3; % 卫星高度
Vr = sqrt(Gravitational * EarthMass / (EarthRadius + H)); 
Vg = Vr * EarthRadius / (EarthRadius + H); % 地面速度

c = 299792458; % 光速
duty_cycle = 0.15; % 占空比
Tr = duty_cycle*(1/1350); % PRF取1100~1600Hz，这里取中间值
PRF = linspace(1100, 1600, 500);
Rt = EarthRadius;
Rs = H + EarthRadius;

int_min = min(floor(2*H*PRF/c)); % 需要注意卫星的高度是 H，斜距必然大于 H
int_max = max(ceil(2*sqrt(Rs^2-Rt^2)*PRF/c)); % 这里假设地球是球形的，斜距不会大于切线长度

% 设置发射窗口在PRF处的范围
frac_min = 0;
frac_max = (Tr) ./ (1 ./ PRF);

figure("name", "斑马图");
for i = int_min:int_max

    R_tx_min = (i + frac_min) * c ./ (2 * PRF); % 发射窗口对应的斜距的下边界，对应的是可用窗口的上边界。
    gamma_cos = (R_tx_min.^2 + Rs^2 - Rt^2) ./ (2 * R_tx_min * Rs);

    % 限制gamma余弦值的范围在[-1, 1]之间，否则取1（角度取0）
    tmp_gamma_cos = (abs(gamma_cos) <= 1) .* gamma_cos;
    gamma_cos = (1 - (abs(gamma_cos) <= 1)) + tmp_gamma_cos;
    gamma1 = acos(gamma_cos);
    incident1 = asin(Rs .* sin(gamma1) ./ Rt);
    incident1 = rad2deg(incident1);

    R_tx_max = (i + frac_max) * c ./ (2 * PRF);  % 发射窗口的上边界，对应的是可用窗口的下边界
    gamma_cos = (R_tx_max.^2 + Rs^2 - Rt^2) ./ (2 * R_tx_max * Rs);
    % 同上
    tmp_gamma_cos = (abs(gamma_cos) <= 1) .* gamma_cos;
    gamma_cos = (1 - (abs(gamma_cos) <= 1)) + tmp_gamma_cos;
    gamma2 = acos(gamma_cos);
    incident2 = asin(Rs .* sin(gamma2) ./ Rt);
    incident2 =  rad2deg(incident2);

    plot(PRF, incident1);
    hold on;
    plot(PRF, incident2);
    hold on;

    % 填充发射窗口
    pic = fill([PRF, fliplr(PRF)], [incident1, fliplr(incident2)], 'b');
    set(pic, 'facealpha', 0.5);
end

% 计算星下点干扰的窗口
% 由于回波的延时大于 1/PRF，所以第n个发射脉冲的星下点回波可能与第n-1（或者小于n-1）个发射脉冲的回波数据重叠
tau_sub_s = 20e-6; % 星下点干扰的脉冲宽度(假设值) 


for i = 0:int_max  %遍历到最大的可能回波时延

    R_rx = (2*H/c + i./PRF)*c/2; % 星下点的斜距下边界
    gamma_cos = (R_rx.^2 + Rs^2 - Rt^2) ./ (2 * R_rx * Rs);
    tmp_gamma_cos = (abs(gamma_cos) <= 1) .* gamma_cos;
    gamma_cos = (1 - (abs(gamma_cos) <= 1)) + tmp_gamma_cos;
    gamma1 = acos(gamma_cos);
    incident1 = asin(Rs .* sin(gamma1) ./ Rt);
    incident1 = rad2deg(incident1);

    R_rx = (2*H/c + i./PRF + tau_sub_s)*c/2; % 星下点的斜距上边界
    gamma_cos = (R_rx.^2 + Rs^2 - Rt^2) ./ (2 * R_rx * Rs);
    tmp_gamma_cos = (abs(gamma_cos) <= 1) .* gamma_cos;
    gamma_cos = (1 - (abs(gamma_cos) <= 1)) + tmp_gamma_cos;
    gamma2 = acos(gamma_cos);
    incident2 = asin(Rs .* sin(gamma2) ./ Rt);
    incident2 = rad2deg(incident2);

    plot(PRF, incident1);
    hold on;
    plot(PRF, incident2);
    hold on;

    % 填充星下点干扰窗口
    pic = fill([PRF, fliplr(PRF)], [incident1, fliplr(incident2)], 'r');
    set(pic, 'facealpha', 0.5);
end

% 各个条带的PRF与入射角
PRF_swath = [1340 1250 1350 1260 1330 1260];
incident_swath_min = deg2rad([20 27.5 34.9 40.9 46.4 50.3]);
incident_swath_max = deg2rad([29 35.6 41.9 47 51.7 55]);
incident_swath = [incident_swath_min;incident_swath_max];

for i = 1:6
    plot(ones(1, length(PRF))*PRF_swath(i), linspace(rad2deg(incident_swath_min(i)), ...
    rad2deg(incident_swath_max(i)), length(PRF)), 'k--');
    hold on;
end

ylim([rad2deg(incident_min), rad2deg(incident_max)]);
xlabel("PRF/Hz");
ylabel("入射角");
grid on;
view([90 -90]); % 旋转视角
title("斑马图");

% 计算各条带对应的测绘宽度
gamma_swath_max = asin(EarthRadius * sin(incident_swath_max) ./ (H + EarthRadius));
beta_swath_max = incident_swath_max - gamma_swath_max;

gamma_swath_min = asin(EarthRadius * sin(incident_swath_min) ./ (H + EarthRadius)); % 下视角
beta_swath_min = incident_swath_min - gamma_swath_min;

Wg_swath = EarthRadius*(beta_swath_max - beta_swath_min);

% 计算各条带对应的距离向天线长度
gamma_swath = gamma_swath_max-gamma_swath_min;
dev_tx = 0.886*lambda./gamma_swath;
beta_swath_max = incident_swath_max - gamma_swath_max;
beta_swath_min = incident_swath_min - gamma_swath_min;
R_swath_max = sqrt((Rs^2 + Rt^2 - 2*Rs*Rt*cos(beta_swath_max)));
R_swath_min = sqrt(Rs^2 + Rt^2 - 2*Rs*Rt*cos(beta_swath_min));

% 为了减少距离模糊，接收波束的3dB宽度应该小于脉冲的地面瞬时覆盖范围
beta_max = incident_max - gamma_max;
R_max = sqrt((Rs^2 + Rt^2 - 2*Rs*Rt*cos(beta_max)));
max_hev_rx = 0.886*lambda*R_max/(c*Tr/(2*tan(incident_max)));

% 单个子孔径应该能够覆盖整个区域
max_lev_rx = 0.886*lambda./(max(gamma_swath_max)-min(gamma_swath_min));

% 计算距离向带宽
max_delta_rg = 1; % 地距分辨率
max_delta_ev = max_delta_rg*sin(incident_swath_min); % 斜距分辨率
Bd = c./(max_delta_ev*2); % 去掉了系数以增大裕度

% 估计方位向子孔径数量
max_delta_az = 1;
max_laz = 2*max_delta_az;
min_B_dop = 0.886*2*Vr/max_laz;
min_Naz = ceil(min_B_dop/1250);

% Naz > min_Naz
Naz = 7;

% 估算合适的方位向子孔径大小，使得uniform PRF 在PRF范围内 1240 ~ 1470 Hz 
PRF_uni = 1350;
Laz_rx = 2*Vr/PRF_uni;
daz_rx = Laz_rx/Naz;

B_dop = 0.886*2*Vr/daz_rx; % 该天线孔径下能够得到的多普勒带宽

% 方位向的发射孔径通过评估aasr得到

% 计算分析模型的aasr

% 设置待观察的发射孔径
laz_tx = 1:0.1:10; 

swath_num = length(PRF_swath);
faz = -B_dop/2:B_dop/2;

mat_faz = repmat(faz, length(laz_tx), 1);
mat_laz = repmat(laz_tx', 1, length(faz));

% 计算所有条带
figure("name","aasr and laz_tx");

for i = 1:swath_num
    prf = PRF_swath(i);
    R_swath = R_swath_min(i);
    G_tx = sinc(mat_laz.*mat_faz/(2*Vr)).^2;
    G_rx = sinc(daz_rx*mat_faz/(2*Vr)).^2;
    G = G_tx.*G_rx;
    % 计算重构滤波器的系数
    P = zeros(Naz, Naz, length(faz));
    H = zeros(Naz, Naz);
    for j = 1:length(faz)
        for k = 1:Naz
            for n = 1:Naz
                H(k, n) = exp(-1j*pi*(Vg/Vr)*((n-1)*daz_rx)^2/(2*lambda*R_swath)-1j*pi*((n-1)*daz_rx)/Vr*(faz(j)+(k-1)*prf));
            end
        end
        P(:, :, j) = inv(H);
    end
    m = 28; % 方位向模糊阶数
    aasr_num = zeros(length(laz_tx), 1);
    aasr_deno = zeros(length(laz_tx), 1);
    for j = 0:m
        G_tmp_tx = sinc((mat_faz+j*prf).*mat_laz/(2*Vr)).^2;
        G_tmp_rx = sinc((mat_faz+j*prf)*daz_rx/(2*Vr)).^2;
        G_tmp = G_tmp_rx.*G_tmp_tx;

        % 计算每个频率的混叠
        sys_reconstruct = zeros(length(laz_tx), length(faz));
        for h = 1:length(faz)
            H = zeros(Naz, Naz);
            for k = 1:Naz
                for n = 1:Naz
                    H(k, n) = exp(-1j*pi*(Vg/Vr)*((n-1)*daz_rx)^2/(2*lambda*R_swath)-1j*pi*((n-1)*daz_rx)/Vr*(faz(h)+(k-1)*prf+j*prf));
                end
            end
            A_per_band = H*P(:,:,h);
            sys_reconstruct(:, h) = ones(length(laz_tx),1)*A_per_band(1,1);
        end
        G_tmp = G_tmp.*abs(sys_reconstruct);
        if(j == 0)
            aasr_deno = 2*trapz(faz, G_tmp, 2);
        else 
            aasr_num = 2*trapz(faz, G_tmp, 2)+aasr_num;
        end
    end
    aasr = 10*log10(aasr_num./aasr_deno);
    plot(laz_tx, aasr);
    hold on;
end

title("aasr and azimuth tx ant length");

figure("name", "最大多普勒频率处的方位向发射增益")

% 单程增益不应该太低
G_tx = sinc(laz_tx.*(B_dop/2)/(2*Vr));
plot(laz_tx, 10*log10(abs(G_tx)));
title("最大多普勒频率处的方位向发射增益");


daz_tx = 3; % 观察图3,4得到的最佳发射孔径大小
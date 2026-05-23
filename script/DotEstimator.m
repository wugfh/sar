classdef DotEstimator < handle
    properties
        points_n
        c
        Vr
        PRF
        Fs
        path
    end

    methods
        function obj = DotEstimator(point_n, c, Vr, PRF, Fs, path)
            obj.points_n = point_n;
            obj.c = c;
            obj.Vr = Vr;
            obj.PRF = PRF;
            obj.Fs = Fs;
            obj.path = path;
        end

        function data_up = upsample(obj, data, N)
            % 二维频域补零上采样，N = [up_az, up_rg]
            [Na, Nr] = size(data);
            data_fft = fftshift(fft2(data));
            tmp = zeros(N(1)*Na, N(2)*Nr);
            r_center = floor(N(1)*Na/2);
            c_center = floor(N(2)*Nr/2);
            r_idx = r_center - floor(Na/2) + 1 : r_center + ceil(Na/2);
            c_idx = c_center - floor(Nr/2) + 1 : c_center + ceil(Nr/2);
            tmp(r_idx, c_idx) = data_fft;
            data_up = ifft2(ifftshift(tmp));
        end

        function irw = get_range_IRW(obj, echo_abs, uprate)
            % 距离向 -3dB IRW
            target = max(echo_abs, [], 1);
            max_val = max(target);
            half_max = max_val / sqrt(2);
            irw_samples = sum(target > half_max);
            irw = irw_samples * obj.c / (2 * uprate * obj.Fs);
        end

        function irw = get_azimuth_IRW(obj, echo_abs, uprate)
            % 方位向 -3dB IRW
            target = max(echo_abs, [], 2)';
            max_val = max(target);
            half_max = max_val / sqrt(2);
            irw_samples = sum(target > half_max);
            irw = irw_samples * obj.Vr / (obj.PRF * uprate);
        end

        function pslr = get_pslr(obj, target)
            % 峰值旁瓣比 (PSLR) in dB
            [pks, locs] = findpeaks(target);
            if isempty(pks)
                pslr = -inf; return;
            end
            [main_val, main_idx] = max(pks);
            main_loc = locs(main_idx);
            sidelobe_pks = pks;
            sidelobe_pks(main_idx) = [];
            if isempty(sidelobe_pks)
                pslr = -inf; return;
            end
            sidelobe_val = max(sidelobe_pks);
            pslr = 20 * log10(sidelobe_val / main_val);
        end

        function islr = get_islr(obj, target)
            % 积分旁瓣比 (ISLR) in dB, target 为功率
            total_power = sum(target);
            [pks, locs] = findpeaks(target);
            if isempty(pks)
                islr = -inf; return;
            end
            [~, idx] = max(pks);
            main_loc = locs(idx);

            % 寻找主瓣两侧最近的谷底
            [~, valley_locs] = findpeaks(-target);
            left_valleys = valley_locs(valley_locs < main_loc);
            right_valleys = valley_locs(valley_locs > main_loc);
            if isempty(left_valleys) || isempty(right_valleys)
                islr = -inf; return;
            end
            left_idx = left_valleys(end);
            right_idx = right_valleys(1);
            main_power = sum(target(left_idx:right_idx));
            side_power = total_power - main_power;
            islr = 10 * log10(side_power / main_power);
        end

        function dot_estimate(obj, image, area, uprate)
            % 分析前 points_n 个最强点目标
            figure('Position', [100 100 400*obj.points_n 800]);
            image_copy = image;
            processed = 0;
            range_res_list = [];

            while processed < obj.points_n
                [~, max_idx] = max(abs(image_copy(:)));
                [max_r, max_c] = ind2sub(size(image_copy), max_idx);

                if (max_r <= area(1)/2 || max_r > size(image_copy,1)-area(1)/2 || ...
                    max_c <= area(2)/2 || max_c > size(image_copy,2)-area(2)/2)
                    image_copy(max_r, max_c) = 0;
                    continue;
                end

                r_start = max_r - floor(area(1)/2);
                r_end   = max_r + floor(area(1)/2);
                c_start = max_c - floor(area(2)/2);
                c_end   = max_c + floor(area(2)/2);
                target_patch = image_copy(r_start:r_end, c_start:c_end);
                target_up = obj.upsample(target_patch, [uprate, uprate]);
                target_abs = abs(target_up);

                dr = [-area(2)/2, area(2)/2] * obj.c / (2 * obj.Fs);
                da = [-area(1)/2, area(1)/2] * obj.Vr / obj.PRF;

                % 距离向
                range_irw = obj.get_range_IRW(target_abs, uprate);
                range_res_list = [range_res_list, range_irw];
                range_prof = max(target_abs, [], 1);
                range_prof = range_prof / max(range_prof);
                x_dr = linspace(dr(1), dr(2), length(range_prof));

                subplot(3, obj.points_n, obj.points_n + processed + 1);
                plot(x_dr, 20*log10(range_prof)); grid on;
                ylim([-30 0]); xlabel('Range (m)'); ylabel('Amplitude (dB)');
                title(sprintf('(%c) R-profile', char('a'+processed)));

                % 方位向
                azimuth_irw = obj.get_azimuth_IRW(target_abs, uprate);
                azimuth_prof = max(target_abs, [], 2)';
                azimuth_prof = azimuth_prof / max(azimuth_prof);
                x_da = linspace(da(1), da(2), length(azimuth_prof));

                subplot(3, obj.points_n, 2*obj.points_n + processed + 1);
                plot(x_da, 20*log10(azimuth_prof)); grid on;
                ylim([-30 0]); xlabel('Azimuth (m)'); ylabel('Amplitude (dB)');
                title(sprintf('(%c) A-profile', char('a'+processed)));

                % 二维图
                image_db = 20*log10(target_abs / max(target_abs(:)));
                subplot(3, obj.points_n, processed + 1);
                imagesc(dr, da, image_db, [-60 0]); axis image; colorbar;
                set(gca, 'YDir', 'normal');
                xlabel('Range (m)'); ylabel('Azimuth (m)');
                title(sprintf('(%c) 2D', char('a'+processed)));

                fprintf('Range IRW: %.4f m\n', range_irw);
                fprintf('Azimuth IRW: %.4f m\n', azimuth_irw);
                fprintf('Range PSLR: %.2f dB\n', obj.get_pslr(range_prof));
                fprintf('Azimuth PSLR: %.2f dB\n', obj.get_pslr(azimuth_prof));

                % 清除区域
                image_copy(r_start:r_end, c_start:c_end) = 0;
                processed = processed + 1;
            end

            save_path = fullfile(obj.path, 'dot_estimate.png');
            saveas(gcf, save_path);
            fprintf('Average range resolution: %.4f m\n', mean(range_res_list));
        end

        function pslr = pslr_estimate(obj, image, area, uprate)
            [~, max_idx] = max(abs(image(:)));
            [max_r, max_c] = ind2sub(size(image), max_idx);
            r_start = max_r - floor(area(1)/2);
            r_end   = max_r + floor(area(1)/2);
            c_start = max_c - floor(area(2)/2);
            c_end   = max_c + floor(area(2)/2);
            if r_start < 1 || r_end > size(image,1) || ...
               c_start < 1 || c_end > size(image,2)
                disp('Maximum point too close to boundary');
                pslr = NaN; return;
            end
            target_patch = image(r_start:r_end, c_start:c_end);
            target_up = obj.upsample(target_patch, [uprate, uprate]);
            azimuth_prof = max(abs(target_up), [], 2)';
            azimuth_prof = azimuth_prof / max(azimuth_prof);
            pslr = obj.get_pslr(azimuth_prof);
        end
    end
end
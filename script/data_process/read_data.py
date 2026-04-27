import os
import struct
import sys
import numpy as np
import argparse

# /home/zenghong/sar/script/data_process/read_data.py

def read_echo(file_name, max_frame, skip_frame):
    """
    Python translation of the MATLAB read_echo function.
    Returns: sig (complex128 ndarray), fcs (1D float64 ndarray),
             frame_time (1D uint32 ndarray), params (dict or None)
    """

    frames = []
    fcs = []
    frame_times = []
    PRF = Tr = Br = f0 = t0 = Fr = None

    total_count = 0
    f = open(file_name, 'rb')
    try:
        while True:
            try:
                filesize = os.fstat(f.fileno()).st_size
                if f.tell() >= filesize:
                    break
            except Exception:
                # Fallback: try to read a byte to detect EOF, rewind if not EOF
                b = f.read(1)
                if not b:
                    break
                f.seek(-1, os.SEEK_CUR)
            frame_begin = f.tell()
            head = f.read(8)
            if len(head) < 8:
                break
            total_count += 1
            if total_count > max_frame + skip_frame:
                break

            head1, head2 = struct.unpack('<II', head)
            if head1 != 0x11FFFF11 or head2 != 0x20230211:
                print(f"Error in Frame {total_count}: incorrect frame head", file=sys.stderr)

            # timer_info (uint64), frame_count (uint32), frame_len (uint32)
            hdr = f.read(8 + 4 + 4)
            if len(hdr) < 16:
                break
            timer_info, frame_count, frame_len = struct.unpack('<QII', hdr)

            # skip frames if requested
            if total_count <= skip_frame:
                # seek to next frame
                try:
                    f.seek(frame_begin + frame_len, os.SEEK_SET)
                except OSError:
                    print("Skip to EOF, no data read.", file=sys.stderr)
                    break
                continue

            # center frequency code at offset 31
            f.seek(frame_begin + 31, os.SEEK_SET)
            b = f.read(1)
            if len(b) < 1:
                break
            flags = b[0]
            msb = bool(flags & 0x80)  # bit 8
            bit7 = bool(flags & 0x40)  # bit 7
            if (not msb) and (not bit7):
                fcs.append(33e9)
            elif (not msb) and bit7:
                fcs.append(37e9)
            elif msb and (not bit7):
                fcs.append(35e9)
            else:
                fcs.append(np.nan)

            # control parameters at offset 116
            f.seek(frame_begin + 116, os.SEEK_SET)
            buf = f.read(4 + 2 + 2 + 2 + 4)  # uint32, uint16, uint16, uint16, uint32 (but note t0 read after)
            if len(buf) >= 14:
                prf_u32 = struct.unpack_from('<I', buf, 0)[0]
                PRF = 2e8 / prf_u32 if prf_u32 != 0 else None
                Tr_code = struct.unpack_from('<H', buf, 4)[0]
                Br_code = struct.unpack_from('<H', buf, 6)[0]
                f0_code = struct.unpack_from('<H', buf, 8)[0]
                # t0 is read as uint32 (next 4 bytes) but MATLAB reads it after f0 with uint32 at same area
                # if not enough bytes in buf, try to read separately
                try:
                    t0_u32 = struct.unpack_from('<I', buf, 10)[0]
                except Exception:
                    # read separately
                    f.seek(frame_begin + 116 + 10, os.SEEK_SET)
                    t0_bytes = f.read(4)
                    if len(t0_bytes) == 4:
                        t0_u32 = struct.unpack('<I', t0_bytes)[0]
                    else:
                        t0_u32 = 0
                Tr = 0.1e-6 * Tr_code
                Br = 0.1e6 * Br_code
                f0 = 1e6 * f0_code
                t0 = 5e-3 * 1e-6 * t0_u32

            # Fr at offset 138
            f.seek(frame_begin + 138, os.SEEK_SET)
            buf = f.read(4)
            if len(buf) == 4:
                Fr_code = struct.unpack('<I', buf)[0]
                Fr = 0.1e6 * Fr_code

            # clock_time at offset 276
            f.seek(frame_begin + 276, os.SEEK_SET)
            buf = f.read(4)
            if len(buf) == 4:
                clock_time = struct.unpack('<I', buf)[0]
                frame_times.append(clock_time)
            else:
                frame_times.append(0)

            # frame data at offset 512
            if frame_len < 512:
                print(f"Frame {total_count} has invalid length {frame_len}, skipped.", file=sys.stderr)
                try:
                    f.seek(frame_begin + frame_len, os.SEEK_SET)
                except OSError:
                    break
                continue

            f.seek(frame_begin + 512, os.SEEK_SET)
            data_bytes = f.read(frame_len - 512)
            expected_int16 = (frame_len - 512) // 2
            if len(data_bytes) < expected_int16 * 2:
                print(f"Frame {total_count} incomplete: read {len(data_bytes)//2}, expected {expected_int16}, discarded.", file=sys.stderr)
                break

            arr = np.frombuffer(data_bytes, dtype='<i2').copy()  # int16 little-endian
            frames.append(arr)

            # move file pointer to end of frame to prepare next header read
            try:
                f.seek(frame_begin + frame_len, os.SEEK_SET)
            except OSError:
                break

            # stop if we've collected required frames
            if len(frames) >= max_frame:
                break

    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        return None, None, None, None

    if len(frames) == 0:
        return None, None, None, None

    # make 2D matrix: each column is a frame (MATLAB used [frames{:}] which makes columns)
    min_len = min(arr.size for arr in frames)
    if min_len == 0:
        return None, None, None, None
    # truncate to min_len to ensure rectangular array
    mat = np.vstack([arr[:min_len] for arr in frames]).T  # shape (min_len, n_frames)

    # form complex samples: pair rows as I/Q (MATLAB: complex(sig(1:2:end,:), sig(2:2:end,:)))
    # Here rows = samples; we pair row 0&1, 2&3, ...
    n_rows = mat.shape[0]
    n_pairs = n_rows // 2
    I = mat[0:2*n_pairs:2, :].astype(np.float64)
    Q = mat[1:2*n_pairs:2, :].astype(np.float64)
    complex_mat = I + 1j * Q  # shape (n_pairs, n_frames)

    # use only V channel: MATLAB did sig = sig(1:2:end, :) after complex formation
    sig = complex_mat[0:complex_mat.shape[0]:2, :].astype(np.complex128)

    fcs_arr = np.array(fcs[:sig.shape[1]], dtype=np.float64)
    frame_time_arr = np.array(frame_times[:sig.shape[1]], dtype=np.uint32)

    params = {
        'Tr': Tr,
        'Br': Br,
        'f0': fcs_arr[0] if len(fcs_arr) > 0 else None,  # use fcs from frame if available
        't0': t0,
        'Fr': Fr,
        'PRF': PRF
    } if len(frames) > 0 else None
    return sig, fcs_arr, frame_time_arr, params


# If run as script for quick test (not required)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("file")
    p.add_argument("--max", type=int, default=1000)
    p.add_argument("--skip", type=int, default=0)
    args = p.parse_args()
    sig, fcs, frame_time, params = read_echo(args.file, args.max, args.skip)
    print("sig shape:", None if sig is None else sig.shape)
    print("len fcs:", None if fcs is None else fcs.shape)
    print("len frame_time:", None if frame_time is None else frame_time.shape)
    print("params:", params)
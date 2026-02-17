import os
import shutil
import argparse
import sys
import warnings
import json
import re
# import tkinter as tk  <-- ELIMINADO PARA SERVIDOR
# from tkinter import filedialog <-- ELIMINADO PARA SERVIDOR
from glob import glob
from contextlib import contextmanager
import csv

# Suppress warnings from libraries
warnings.filterwarnings("ignore")

try:
    import mutagen
    from mutagen.mp3 import MP3
    from mutagen.easyid3 import EasyID3
    from mutagen.id3 import ID3NoHeaderError
    from pydub import AudioSegment
    import librosa
    
    # AJUSTE PARA SERVIDOR: Usar backend sin pantalla antes de importar pyplot
    import matplotlib
    matplotlib.use('Agg') 
    import librosa.display
    import matplotlib.pyplot as plt
    
    import numpy as np
    from tqdm import tqdm
    import scipy.stats
    import colorama
    from colorama import Fore, Back, Style
except ImportError as e:
    print(f"Error crítico: Falta la librería {e.name}.")
    print("Por favor, ejecuta: pip install -r requirements.txt")
    sys.exit(1)

# Variables globales para control de errores
FFMPEG_ERROR_SHOWN = False
CACHE_FILE = "audio_analysis_cache.json"

print(f"{Fore.CYAN}Archivo de caché: {os.path.abspath(CACHE_FILE)}{Style.RESET_ALL}")

@contextmanager
def suppress_stderr():
    """Silencia las salidas de error de bajo nivel (C libraries) redirigiendo stderr a null."""
    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        with open(os.devnull, 'w') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(saved_stderr_fd, stderr_fd)
                os.close(saved_stderr_fd)
    except Exception:
        yield

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(cache):
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=4)
    except Exception as e:
        print(f"Error guardando caché: {e}")

colorama.init(autoreset=True)

# Perfiles de tonalidad (Krumhansl-Schmuckler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

CAMELOT_MAJOR = {
    0: '8B',  1: '3B',  2: '10B', 3: '5B',  4: '12B', 5: '7B',
    6: '2B',  7: '9B',  8: '4B',  9: '11B', 10: '6B', 11: '1B'
}
CAMELOT_MINOR = {
    0: '5A',  1: '12A', 2: '7A',  3: '2A',  4: '9A',  5: '4A',
    6: '11A', 7: '6A',  8: '1A',  9: '8A',  10: '3A', 11: '10A'
}
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def estimate_key(y, sr):
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.sum(chroma, axis=1)
        
        corrs_major = []
        corrs_minor = []
        
        for i in range(12):
            shifted_major = np.roll(MAJOR_PROFILE, i)
            shifted_minor = np.roll(MINOR_PROFILE, i)
            corrs_major.append(np.corrcoef(chroma_vals, shifted_major)[0, 1])
            corrs_minor.append(np.corrcoef(chroma_vals, shifted_minor)[0, 1])
            
        best_major_idx = np.argmax(corrs_major)
        best_minor_idx = np.argmax(corrs_minor)
        
        max_major = corrs_major[best_major_idx]
        max_minor = corrs_minor[best_minor_idx]
        
        if max_major > max_minor:
            key_idx = best_major_idx
            scale = 'Major'
            camelot = CAMELOT_MAJOR[key_idx]
        else:
            key_idx = best_minor_idx
            scale = 'Minor'
            camelot = CAMELOT_MINOR[key_idx]
            
        return f"{NOTE_NAMES[key_idx]} {scale}", camelot
    except Exception as e:
        return "Unknown", "---"

def get_audio_info(filepath, calculate_gain=True):
    global FFMPEG_ERROR_SHOWN
    
    info = {
        'path': filepath,
        'filename': os.path.basename(filepath),
        'folder': os.path.dirname(filepath),
        'bitrate': 0,
        'samplerate': 0,
        'duration': 0,
        'dbfs': -99.0,
        'key': 'Unknown',
        'camelot': '---',
        'bpm': 0,
        'artist': 'Unknown',
        'title': 'Unknown',
        'album': 'Unknown'
    }

    try:
        filepath = os.path.normpath(filepath)
        audio = mutagen.File(filepath)
        
        if audio:
            if hasattr(audio.info, 'length'):
                info['duration'] = audio.info.length
            if hasattr(audio.info, 'bitrate') and audio.info.bitrate:
                info['bitrate'] = audio.info.bitrate
            if hasattr(audio.info, 'sample_rate'):
                info['samplerate'] = audio.info.sample_rate

            def get_tag(keys):
                for k in keys:
                    if hasattr(audio, 'tags') and audio.tags:
                        if k in audio.tags: return str(audio.tags[k][0])
                        if k.upper() in audio.tags: return str(audio.tags[k.upper()][0])
                    if k in audio: return str(audio[k][0])
                return None

            artist = get_tag(['artist', 'TPE1', 'IART'])
            title = get_tag(['title', 'TIT2', 'INAM'])
            album = get_tag(['album', 'TALB', 'IPRD'])

            if artist: info['artist'] = artist
            if title: info['title'] = title
            if album: info['album'] = album

        if calculate_gain:
            try:
                with suppress_stderr():
                    seg = AudioSegment.from_file(filepath)
                info['dbfs'] = seg.dBFS
                
                if info['bitrate'] == 0 and info['duration'] > 0:
                    file_size = os.path.getsize(filepath)
                    info['bitrate'] = (file_size * 8) / info['duration']
                
                duration = info['duration']
                offset_sec = max(0, duration/2 - 30)
                
                print(f"   [{os.path.basename(filepath)}] Analizando fragmento...")
                
                start_ms = int(offset_sec * 1000)
                end_ms = start_ms + 30000
                chunk = seg[start_ms:end_ms]
                
                samples = np.array(chunk.get_array_of_samples())
                if chunk.channels == 2:
                    samples = samples.reshape((-1, 2))
                    y = samples.mean(axis=1)
                else:
                    y = samples
                
                y = y.astype(np.float32) / 32768.0
                sr = chunk.frame_rate
                
                if len(y.shape) > 1: y = y.flatten()
                y = np.ascontiguousarray(y)
                
                try:
                    key_name, camelot_code = estimate_key(y, sr)
                except Exception as e:
                    print(f"   Error estimando tonalidad: {e}")
                    key_name, camelot_code = 'Unknown', '---'
                
                info['key'] = key_name
                info['camelot'] = camelot_code

                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    if isinstance(tempo, np.ndarray):
                        tempo = tempo[0] if tempo.size > 0 else 0
                    info['bpm'] = round(tempo)
                except Exception as e:
                     print(f"   Error estimando BPM: {e}")
                     info['bpm'] = 0
                
            except (FileNotFoundError, OSError) as e:
                pass
            except Exception as e:
                pass

    except Exception as e:
        print(f"\nError leyendo {os.path.basename(filepath)}: {e}")
    
    return info

def scan_directory(directories, calculate_gain=True, callback=None):
    files = []
    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
    audio_files = []
    
    if isinstance(directories, str):
        directories = [directories]
    
    for directory in directories:
        for ext in extensions:
            pattern = os.path.join(directory, '**', ext)
            found = glob(pattern, recursive=True)
            audio_files.extend(found)
    
    audio_files = list(set([os.path.abspath(f) for f in audio_files]))
    print(Fore.CYAN + f"Encontrados {len(audio_files)} archivos." + Style.RESET_ALL)
    
    cache = load_cache()
    files_to_analyze = []
    cached_count = 0
    
    for f_path in audio_files:
        try:
            mtime = os.path.getmtime(f_path)
            size = os.path.getsize(f_path)
            
            if f_path in cache:
                cached_data = cache[f_path]
                info = cached_data.get('info', {})
                if cached_data.get('mtime') == mtime and cached_data.get('size') == size:
                    if calculate_gain and info.get('dbfs', -99.0) == -99.0:
                        files_to_analyze.append(f_path)
                    else:
                        files.append(info)
                        cached_count += 1
                    continue
            files_to_analyze.append(f_path)
        except OSError:
            pass

    if files_to_analyze:
        print(Fore.CYAN + f"Analizando {len(files_to_analyze)} archivos nuevos..." + Style.RESET_ALL)
        # En servidor web tqdm puede ensuciar los logs, pero lo dejamos opcional
        iterator = tqdm(files_to_analyze) if sys.stdout.isatty() else files_to_analyze
        for f in iterator:
            if callback: callback(f)
            info = get_audio_info(f, calculate_gain)
            files.append(info)
            try:
                cache[f] = {
                    'mtime': os.path.getmtime(f),
                    'size': os.path.getsize(f),
                    'info': info
                }
            except: pass
        save_cache(cache)
        
    return files

def print_report(files):
    sorted_files = sorted(files, key=lambda x: (x['bitrate'], x['dbfs']), reverse=True)
    return sorted_files

def export_csv(files, filename="reporte_audio.csv"):
    keys = ['filename', 'artist', 'title', 'album', 'bitrate', 'samplerate', 'duration', 'dbfs', 'bpm', 'key', 'camelot', 'path']
    try:
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for file in files:
                row = {k: file.get(k, '') for k in keys}
                writer.writerow(row)
    except Exception as e:
        print(Fore.RED + f"Error exportando CSV: {e}" + Style.RESET_ALL)

# --- FUNCIONES MODIFICADAS PARA WEB ---

def select_folder():
    """Versión segura para servidor (sin ventana emergente)."""
    print("ADVERTENCIA: Selección de carpeta interactiva no disponible en servidor web.")
    return None

def select_multiple_folders():
    """Versión segura para servidor (sin ventana emergente)."""
    print("ADVERTENCIA: Selección de carpetas interactiva no disponible en servidor web.")
    return []

# --------------------------------------

def main():
    # En entorno web, main() raramente se llama, pero lo mantenemos seguro.
    parser = argparse.ArgumentParser(description="Herramienta de análisis audio")
    parser.add_argument('--path', help='Directorio a escanear')
    args = parser.parse_args()

    target_paths = [args.path] if args.path else []
    
    if not target_paths:
        # En servidor no podemos preguntar interactivamente con input() si no hay terminal
        if not sys.stdin.isatty():
            print("Modo no interactivo: Se requiere argumento --path")
            return
            
        print("Modo interactivo (CLI Local)")
        # Aquí podrías poner lógica de input() solo si estás en local
        target_paths = ['.']

    files = scan_directory(target_paths)
    print_report(files)

if __name__ == "__main__":
    main()
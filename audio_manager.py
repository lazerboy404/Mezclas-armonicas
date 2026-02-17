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

# Silenciar advertencias de librerías
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
    print("Por favor, confirma que esté en requirements.txt")
    sys.exit(1)

# Variables globales y configuración
FFMPEG_ERROR_SHOWN = False
CACHE_FILE = "audio_analysis_cache.json"

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

# Inicializar colorama
colorama.init(autoreset=True)

# Perfiles de tonalidad (Krumhansl-Schmuckler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Mapeo a Rueda de Camelot
CAMELOT_MAJOR = {
    0: '8B',  1: '3B',  2: '10B', 3: '5B',  4: '12B', 5: '7B',
    6: '2B',  7: '9B',  8: '4B',  9: '11B', 10: '6B', 11: '1B'
}
CAMELOT_MINOR = {
    0: '5A',  1: '12A', 2: '7A',  3: '2A',  4: '9A',  5: '4A',
    6: '11A', 7: '6A',  8: '1A',  9: '8A',  10: '3A', 11: '10A'
}
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

# --- FUNCIONES DE ANÁLISIS ---

def estimate_key(y, sr):
    """Estima la tonalidad y retorna (Nota, Escala, Camelot)."""
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
        
        if corrs_major[best_major_idx] > corrs_minor[best_minor_idx]:
            return f"{NOTE_NAMES[best_major_idx]} Major", CAMELOT_MAJOR[best_major_idx]
        else:
            return f"{NOTE_NAMES[best_minor_idx]} Minor", CAMELOT_MINOR[best_minor_idx]
    except Exception:
        return "Unknown", "---"

def get_audio_info(filepath, calculate_gain=True):
    """Obtiene información técnica, ganancia, BPM y Key del archivo."""
    info = {
        'path': filepath,
        'filename': os.path.basename(filepath),
        'bitrate': 0,
        'duration': 0,
        'dbfs': -99.0,
        'key': 'Unknown',
        'camelot': '---',
        'bpm': 0
    }

    try:
        audio = mutagen.File(filepath)
        if audio and hasattr(audio.info, 'length'):
            info['duration'] = audio.info.length
        
        if calculate_gain:
            try:
                with suppress_stderr():
                    seg = AudioSegment.from_file(filepath)
                info['dbfs'] = seg.dBFS
                
                # Análisis de fragmento de 30 segundos para Key y BPM
                y, sr = librosa.load(filepath, sr=None, offset=max(0, info['duration']/2 - 15), duration=30)
                
                key_name, camelot_code = estimate_key(y, sr)
                info['key'] = key_name
                info['camelot'] = camelot_code

                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                info['bpm'] = round(float(tempo)) if isinstance(tempo, (float, int, np.ndarray)) else 0
                
            except Exception:
                pass
    except Exception as e:
        print(f"Error procesando {os.path.basename(filepath)}: {e}")
    
    return info

# --- LÓGICA DE COMPATIBILIDAD CAMELOT (RESTAURADA) ---

def get_compatible_keys(camelot_code):
    """Retorna lista de claves compatibles según la Rueda de Camelot."""
    if not camelot_code or camelot_code == '---':
        return []
    
    try:
        match = re.match(r"(\d+)([AB])", camelot_code)
        if not match:
            return []
            
        number = int(match.group(1))
        letter = match.group(2)
    except:
        return []
        
    compatible = [camelot_code]
    
    # Relativa (A <-> B)
    other_letter = 'B' if letter == 'A' else 'A'
    compatible.append(f"{number}{other_letter}")
    
    # +/- 1 (Mismo anillo)
    plus_one = number + 1 if number < 12 else 1
    minus_one = number - 1 if number > 1 else 12
    
    compatible.append(f"{plus_one}{letter}")
    compatible.append(f"{minus_one}{letter}")
    
    return compatible

def parse_camelot_code(code):
    """Separa el número y la letra del código Camelot."""
    try:
        match = re.match(r"(\d+)([AB])", code)
        if match:
            return int(match.group(1)), match.group(2)
    except:
        pass
    return None, None

# --- FUNCIONES DE SOPORTE Y NEUTRALIZACIÓN ---

def load_cache():
    return json.load(open(CACHE_FILE, 'r', encoding='utf-8')) if os.path.exists(CACHE_FILE) else {}

def save_cache(cache):
    json.dump(cache, open(CACHE_FILE, 'w', encoding='utf-8'), indent=4)

def select_folder(): return None
def select_multiple_folders(): return []

def main():
    print("Módulo audio_manager cargado correctamente para producción.")

if __name__ == "__main__":
    main()
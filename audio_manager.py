import os
import shutil
import argparse
import sys
import warnings
import json
import re
import tkinter as tk
from tkinter import filedialog
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
    print("Nota: También necesitas tener ffmpeg instalado en tu sistema.")
    sys.exit(1)

# Variables globales para control de errores
FFMPEG_ERROR_SHOWN = False
CACHE_FILE = "audio_analysis_cache.json"

print(f"{Fore.CYAN}Archivo de caché: {os.path.abspath(CACHE_FILE)}{Style.RESET_ALL}")

@contextmanager
def suppress_stderr():
    """Silencia las salidas de error de bajo nivel (C libraries) redirigiendo stderr a null."""
    try:
        # Guardar stderr original
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        
        # Abrir null device
        with open(os.devnull, 'w') as devnull:
            # Redirigir stderr a null
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                # Restaurar
                os.dup2(saved_stderr_fd, stderr_fd)
                os.close(saved_stderr_fd)
    except Exception:
        # Si falla (ej. entorno sin fileno), solo ejecutar código
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

# Inicializar colorama
colorama.init(autoreset=True)

# Perfiles de tonalidad (Krumhansl-Schmuckler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Mapeo a Camelot Wheel
CAMELOT_MAJOR = {
    0: '8B',  1: '3B',  2: '10B', 3: '5B',  4: '12B', 5: '7B',
    6: '2B',  7: '9B',  8: '4B',  9: '11B', 10: '6B', 11: '1B'
}
CAMELOT_MINOR = {
    0: '5A',  1: '12A', 2: '7A',  3: '2A',  4: '9A',  5: '4A',
    6: '11A', 7: '6A',  8: '1A',  9: '8A',  10: '3A', 11: '10A'
}
# Notas para visualización (0=C, 1=C#, etc)
NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def estimate_key(y, sr):
    """Estima la tonalidad y retorna (Nota, Escala, Camelot)."""
    try:
        # Extraer chroma
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_vals = np.sum(chroma, axis=1)
        
        # Correlación con perfiles mayores y menores
        corrs_major = []
        corrs_minor = []
        
        for i in range(12):
            # Rotar perfil para probar cada tónica
            shifted_major = np.roll(MAJOR_PROFILE, i)
            shifted_minor = np.roll(MINOR_PROFILE, i)
            
            # Correlación de Pearson
            corrs_major.append(np.corrcoef(chroma_vals, shifted_major)[0, 1])
            corrs_minor.append(np.corrcoef(chroma_vals, shifted_minor)[0, 1])
            
        # Encontrar el mejor match
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
    """Obtiene información técnica y de ganancia del archivo."""
    global FFMPEG_ERROR_SHOWN
    
    info = {
        'path': filepath,
        'filename': os.path.basename(filepath),
        'folder': os.path.dirname(filepath),  # Carpeta de origen
        'bitrate': 0,
        'samplerate': 0,
        'duration': 0,
        'dbfs': -99.0, # Silencio por defecto
        'key': 'Unknown',
        'camelot': '---',
        'bpm': 0,
        'artist': 'Unknown',
        'title': 'Unknown',
        'album': 'Unknown'
    }

    try:
        # Normalizar ruta
        filepath = os.path.normpath(filepath)
        
        # 1. Metadatos y calidad técnica (usando mutagen.File para soporte genérico)
        audio = mutagen.File(filepath)
        
        if audio:
            # Duración
            if hasattr(audio.info, 'length'):
                info['duration'] = audio.info.length
            
            # Bitrate
            if hasattr(audio.info, 'bitrate') and audio.info.bitrate:
                info['bitrate'] = audio.info.bitrate
            
            # Sample Rate
            if hasattr(audio.info, 'sample_rate'):
                info['samplerate'] = audio.info.sample_rate

            # Extracción de Tags (híbrido para soportar ID3, Vorbis, etc.)
            def get_tag(keys):
                for k in keys:
                    # Intenta acceder como diccionario (MP3/FLAC/Ogg)
                    if hasattr(audio, 'tags') and audio.tags:
                        if k in audio.tags: return str(audio.tags[k][0])
                        if k.upper() in audio.tags: return str(audio.tags[k.upper()][0])
                    # Intenta acceso directo (algunos formatos)
                    if k in audio: return str(audio[k][0])
                return None

            # Claves comunes: [Standard Name, ID3 Frame]
            artist = get_tag(['artist', 'TPE1', 'IART'])
            title = get_tag(['title', 'TIT2', 'INAM'])
            album = get_tag(['album', 'TALB', 'IPRD'])

            if artist: info['artist'] = artist
            if title: info['title'] = title
            if album: info['album'] = album

        # 2. Ganancia y Key (requiere decodificar con FFmpeg/Pydub)
        if calculate_gain:
            try:
                # Pydub auto-detecta formato (MP3, WAV, FLAC, etc.)
                # Suprimir advertencias de stderr
                with suppress_stderr():
                    seg = AudioSegment.from_file(filepath)
                info['dbfs'] = seg.dBFS
                
                # Si el bitrate no se detectó antes (ej. WAV), calcularlo aprox
                if info['bitrate'] == 0 and info['duration'] > 0:
                    file_size = os.path.getsize(filepath)
                    info['bitrate'] = (file_size * 8) / info['duration']
                
                # Librosa para Key y BPM (Reutilizar pydub segment para evitar recargar)
                duration = info['duration']
                offset_sec = max(0, duration/2 - 30)
                
                print(f"   [{os.path.basename(filepath)}] Analizando fragmento de audio...")
                
                # Extraer fragmento de 30s desde pydub
                start_ms = int(offset_sec * 1000)
                end_ms = start_ms + 30000
                chunk = seg[start_ms:end_ms]
                
                # Convertir a numpy array float32 normalizado
                # pydub usa int16 por defecto, necesitamos float32 [-1, 1] para librosa
                samples = np.array(chunk.get_array_of_samples())
                
                if chunk.channels == 2:
                    samples = samples.reshape((-1, 2))
                    # Promedio a mono
                    y = samples.mean(axis=1)
                else:
                    y = samples
                
                # Normalizar (dividir por max int16)
                y = y.astype(np.float32) / 32768.0
                sr = chunk.frame_rate
                
                # Asegurar que y sea un array 1D contiguo para librosa
                if len(y.shape) > 1:
                     y = y.flatten()
                y = np.ascontiguousarray(y)
                
                # Key
                print("   Estimando tonalidad...")
                try:
                    # Usar signal.alarm o thread para timeout en sistemas UNIX, pero en Windows es complejo.
                    # En su lugar, envolvemos en try/catch genérico para atrapar errores de numpy/scipy
                    key_name, camelot_code = estimate_key(y, sr)
                except Exception as e:
                    print(f"   Error estimando tonalidad: {e}")
                    key_name, camelot_code = 'Unknown', '---'
                
                info['key'] = key_name
                info['camelot'] = camelot_code

                # BPM (Tempo)
                print("   Estimando BPM...")
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    if isinstance(tempo, np.ndarray):
                        tempo = tempo[0] if tempo.size > 0 else 0
                    info['bpm'] = round(tempo)
                except Exception as e:
                     print(f"   Error estimando BPM: {e}")
                     info['bpm'] = 0
                
                print(f"   Análisis completado: Key={camelot_code}, BPM={info['bpm']}")

            except (FileNotFoundError, OSError) as e:
                if "The system cannot find the file specified" in str(e) or "WinError 2" in str(e):
                    if not FFMPEG_ERROR_SHOWN:
                        print("\n" + Fore.RED + "!"*60)
                        print("AVISO CRÍTICO: No se encontró FFmpeg instalado.")
                        print("!"*60 + "\n" + Style.RESET_ALL)
                        FFMPEG_ERROR_SHOWN = True
                else:
                    pass
            except Exception as e:
                pass

    except Exception as e:
        print(f"\nError leyendo {os.path.basename(filepath)}: {e}")
    
    return info

def scan_directory(directories, calculate_gain=True, callback=None):
    """Escanea uno o múltiples directorios buscando archivos de audio."""
    files = []
    extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a']
    audio_files = []
    
    # Asegurar que directories sea una lista
    if isinstance(directories, str):
        directories = [directories]
    
    # Búsqueda recursiva en cada directorio para cada extensión
    for directory in directories:
        for ext in extensions:
            # glob recursivo requiere pattern '**/*.ext' y recursive=True
            pattern = os.path.join(directory, '**', ext)
            found = glob(pattern, recursive=True)
            audio_files.extend(found)
    
    # Eliminar duplicados y normalizar rutas a absolutas
    audio_files = list(set([os.path.abspath(f) for f in audio_files]))
    
    print(Fore.CYAN + f"Encontrados {len(audio_files)} archivos de audio en {len(directories)} carpetas." + Style.RESET_ALL)
    
    # Cargar caché
    cache = load_cache()
    files_to_analyze = []
    cached_count = 0
    
    for f_path in audio_files:
        try:
            mtime = os.path.getmtime(f_path)
            size = os.path.getsize(f_path)
            
            # Verificar si está en caché y no ha cambiado
            if f_path in cache:
                cached_data = cache[f_path]
                info = cached_data.get('info', {})
                
                if cached_data.get('mtime') == mtime and cached_data.get('size') == size:
                    # Si se requiere ganancia y la caché no la tiene (es -99.0), re-analizar
                    if calculate_gain and info.get('dbfs', -99.0) == -99.0:
                        files_to_analyze.append(f_path)
                    else:
                        files.append(info)
                        cached_count += 1
                    continue
            
            # Si no, añadir a lista para analizar
            files_to_analyze.append(f_path)
            
        except OSError:
            pass

    if cached_count > 0:
        print(Fore.GREEN + f"Usando {cached_count} archivos desde caché (sin cambios)." + Style.RESET_ALL)
    
    if files_to_analyze:
        print(Fore.CYAN + f"Analizando {len(files_to_analyze)} archivos nuevos o modificados..." + Style.RESET_ALL)
        pbar = tqdm(files_to_analyze)
        for f in pbar:
            if callback:
                callback(f)
            pbar.set_description(f"Analizando: {os.path.basename(f)[:20]}")
            info = get_audio_info(f, calculate_gain)
            files.append(info)
            
            # Actualizar caché
            try:
                cache[f] = {
                    'mtime': os.path.getmtime(f),
                    'size': os.path.getsize(f),
                    'info': info
                }
            except:
                pass
        
        # Guardar caché actualizado
        save_cache(cache)
    elif len(audio_files) > 0:
        print(Fore.GREEN + "Todos los archivos están actualizados." + Style.RESET_ALL)
        
    return files

def print_report(files):
    """Imprime reporte ordenado por calidad (bitrate)."""
    # Ordenar por bitrate descendente (mejor calidad primero) y luego ganancia
    sorted_files = sorted(files, key=lambda x: (x['bitrate'], x['dbfs']), reverse=True)
    
    print("\n" + Fore.CYAN + Style.BRIGHT + "="*120)
    print(f"{'Archivo':<30} | {'Bitrate':<10} | {'BPM':<5} | {'Ganancia':<10} | {'Key':<15} | {'Camelot':<8}")
    print("="*120 + Style.RESET_ALL)
    
    for f in sorted_files:
        filename = (f['filename'][:27] + '..') if len(f['filename']) > 27 else f['filename']
        
        # Colorear según calidad
        bitrate_kbps = f['bitrate']/1000
        if bitrate_kbps >= 320:
            color = Fore.GREEN
        elif bitrate_kbps < 192:
            color = Fore.RED
        else:
            color = Fore.WHITE

        # Colorear Camelot
        camelot_color = Fore.YELLOW if f['camelot'] != '---' else Fore.RED
            
        print(f"{color}{filename:<30} | {bitrate_kbps:.0f} kbps   | {f['bpm']:<5} | {f['dbfs']:.2f} dB   | {Fore.MAGENTA}{f['key']:<15}{color} | {camelot_color}{f['camelot']:<8}{Style.RESET_ALL}")
    print(Fore.CYAN + "="*120 + Style.RESET_ALL)
    return sorted_files

def export_csv(files, filename="reporte_audio.csv"):
    """Exporta los resultados a un archivo CSV."""
    keys = ['filename', 'artist', 'title', 'album', 'bitrate', 'samplerate', 'duration', 'dbfs', 'bpm', 'key', 'camelot', 'path']
    try:
        print(f"\nExportando resultados a {filename}...")
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for file in files:
                # Filtrar diccionario
                row = {k: file.get(k, '') for k in keys}
                writer.writerow(row)
        print(Fore.GREEN + f"Reporte exportado exitosamente a {filename}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Error exportando CSV: {e}" + Style.RESET_ALL)

def normalize_gain(files, target_dbfs=-20.0):
    """Normaliza la ganancia de los archivos."""
    print(Fore.CYAN + f"\nIniciando normalización a {target_dbfs} dBFS..." + Style.RESET_ALL)
    print(Fore.YELLOW + "ADVERTENCIA: Esto re-codificará los archivos MP3, lo que puede causar una leve pérdida de calidad." + Style.RESET_ALL)
    confirm = input("¿Desea continuar? (s/n): ")
    if confirm.lower() != 's':
        return

    output_dir = "normalized_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for f in tqdm(files):
        try:
            seg = AudioSegment.from_mp3(f['path'])
            change_in_dBFS = target_dbfs - seg.dBFS
            normalized_sound = seg.apply_gain(change_in_dBFS)
            
            # Exportar
            out_path = os.path.join(output_dir, f['filename'])
            
            # Detectar formato y configurar exportación
            ext = os.path.splitext(f['filename'])[1].lower().replace('.', '')
            export_kwargs = {'format': ext}
            
            # Solo para MP3 intentamos mantener el bitrate original
            if ext == 'mp3' and f['bitrate'] > 0:
                export_kwargs['bitrate'] = f"{int(f['bitrate']/1000)}k"
            
            normalized_sound.export(out_path, **export_kwargs)
            print(f"Normalizado: {f['filename']} ({change_in_dBFS:+.2f} dB)")
        except Exception as e:
            print(f"Error normalizando {f['filename']}: {e}")
            
    print(f"Archivos normalizados guardados en carpeta '{output_dir}'")

def organize_files(files):
    """Organiza archivos en carpetas Artista/Album."""
    print("\nOrganizando archivos...")
    output_dir = "organized_music"
    
    for f in tqdm(files):
        try:
            # Limpiar nombres de carpetas
            artist = "".join([c for c in f['artist'] if c.isalpha() or c.isdigit() or c==' ']).strip()
            album = "".join([c for c in f['album'] if c.isalpha() or c.isdigit() or c==' ']).strip()
            
            target_path = os.path.join(output_dir, artist, album)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
                
            shutil.copy2(f['path'], os.path.join(target_path, f['filename']))
        except Exception as e:
            print(f"Error organizando {f['filename']}: {e}")
            
    print(f"Archivos organizados en '{output_dir}'")

def show_spectrum(filepath):
    """Muestra el espectro de frecuencia."""
    print(f"\nGenerando espectro para: {filepath}")
    try:
        # Cargar solo 60 segundos para agilizar
        y, sr = librosa.load(filepath, duration=60)
        
        plt.figure(figsize=(12, 8))
        
        # Waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title(f'Forma de Onda (Primeros 60s) - {os.path.basename(filepath)}')
        
        # Spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Espectrograma')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generando espectro: {e}")

def get_compatible_keys(camelot_code):
    """Retorna lista de claves compatibles según la Rueda de Camelot."""
    if camelot_code == '---':
        return []
    
    try:
        # Extraer número y letra
        if len(camelot_code) == 2:
            number = int(camelot_code[0])
            letter = camelot_code[1]
        elif len(camelot_code) == 3:
            number = int(camelot_code[:2])
            letter = camelot_code[2]
        else:
            return []
    except:
        return []
        
    compatible = []
    
    # 1. Misma clave
    compatible.append(camelot_code)
    
    # 2. Relativa (Cambio de anillo A<->B)
    other_letter = 'B' if letter == 'A' else 'A'
    compatible.append(f"{number}{other_letter}")
    
    # 3. +/- 1 Hora (Mismo anillo)
    plus_one = number + 1 if number < 12 else 1
    minus_one = number - 1 if number > 1 else 12
    
    compatible.append(f"{plus_one}{letter}")
    compatible.append(f"{minus_one}{letter}")
    
    return compatible

def parse_camelot_code(code):
    """Parsea el código Camelot (ej. '8A') a tupla (numero, letra)."""
    try:
        if len(code) == 2:
            return int(code[0]), code[1]
        elif len(code) == 3:
            return int(code[:2]), code[2]
    except:
        pass
    return None, None

def suggest_mixes(files):
    """Sugiere mezclas armónicas para cada canción."""
    print(Fore.CYAN + "\n=== Sugerencias de Mezcla Armónica (Camelot Wheel) ===" + Style.RESET_ALL)
    
    # Preguntar por carpetas externas
    print("\n¿Deseas incluir otras carpetas para buscar mezclas? (s/n)")
    use_external = input("Opción: ").lower().strip() == 's'
    
    external_files = []
    if use_external:
        print("\n¿Deseas agregar una sola carpeta o múltiples carpetas?")
        print("1. Una sola carpeta")
        print("2. Múltiples carpetas")
        ext_choice = input("Opción (1/2): ").strip()
        
        if ext_choice == '2':
            print("\nSelecciona las carpetas externas para comparar...")
            ext_paths = select_multiple_folders()
            if ext_paths:
                print(f"Escaneando {len(ext_paths)} carpetas externas...")
                for path in ext_paths:
                    print(f"  - {path}")
                # Reutilizamos scan_directory (usa caché automáticamente)
                external_files = scan_directory(ext_paths, calculate_gain=True)
        else:
            print("\nSelecciona la carpeta externa para comparar...")
            ext_path = select_folder()
            if ext_path:
                print(f"Escaneando carpeta externa: {ext_path}")
                # Reutilizamos scan_directory (usa caché automáticamente)
                external_files = scan_directory(ext_path, calculate_gain=True)
    
    # Filtrar solo archivos con clave válida (Originales)
    valid_files = [f for f in files if f['camelot'] != '---']
    
    # Candidatos = Originales + Externos (Filtrados)
    valid_candidates = valid_files + [f for f in external_files if f['camelot'] != '---']
    
    if not valid_files:
        print("No se detectaron tonalidades válidas en la carpeta principal para generar sugerencias.")
        return

    # Ordenar por BPM para facilitar lectura
    valid_files.sort(key=lambda x: x['bpm'])

    for f in valid_files:
        print(f"\n{Fore.YELLOW}Para: {f['filename']} ({f['camelot']} - {f['bpm']} BPM){Style.RESET_ALL}")
        
        compatible_keys = get_compatible_keys(f['camelot'])
        matches = []
        
        for candidate in valid_candidates:
            if candidate['path'] == f['path']: continue # Skip self
            
            if candidate['camelot'] in compatible_keys:
                # Calcular prioridad de armonía (índice en compatible_keys: 0=Exacto, 1=Relativa, 2/3=Adyacente)
                # compatible_keys = [Misma, Relativa, +1, -1]
                try:
                    priority = compatible_keys.index(candidate['camelot'])
                except ValueError:
                    priority = 99
                    
                # Calcular diferencia de BPM
                bpm_diff = abs(candidate['bpm'] - f['bpm'])
                matches.append((candidate, bpm_diff, priority))
        
        # Ordenar primero por Prioridad (Armonía) y luego por diferencia de BPM
        matches.sort(key=lambda x: (x[2], x[1]))
        
        if not matches:
            print(f"  {Fore.RED}No se encontraron canciones compatibles.{Style.RESET_ALL}")
        else:
            for match, diff, prio in matches:
                # Color code BPM difference
                # Verde: <5% diff, Amarillo: <10%, Rojo: >10% (aprox)
                bpm_color = Fore.GREEN if diff <= 5 else (Fore.YELLOW if diff <= 10 else Fore.RED)
                
                # Clasificación detallada
                mix_label = ""
                mix_desc = ""
                
                curr_num, curr_let = parse_camelot_code(f['camelot'])
                cand_num, cand_let = parse_camelot_code(match['camelot'])
                
                if curr_num is not None and cand_num is not None:
                    # 1. PERFECT
                    if f['camelot'] == match['camelot']:
                        mix_label = f"{Fore.GREEN}[PERFECT]{Style.RESET_ALL}"
                        mix_desc = "Misma tonalidad - (Mantiene la vibra)"
                    
                    # 2. HARMONIC (Mismo número, distinta letra)
                    elif curr_num == cand_num and curr_let != cand_let:
                        mix_label = f"{Fore.YELLOW}[HARMONIC]{Style.RESET_ALL}"
                        if curr_let == 'B' and cand_let == 'A':
                            mix_desc = "Tonalidad relativa - (Cambio de Escala: De Alegre a Serio/Profundo)"
                        elif curr_let == 'A' and cand_let == 'B':
                            mix_desc = "Tonalidad relativa - (Cambio de Escala: De Serio a Alegre/Brillante)"
                    
                    # 3. ENERGY (+/- 1)
                    else:
                        # Subida (+1) -> Next = Current + 1 OR (Current=12, Next=1)
                        is_up = (cand_num == curr_num + 1) or (curr_num == 12 and cand_num == 1)
                        # Bajada (-1) -> Next = Current - 1 OR (Current=1, Next=12)
                        is_down = (cand_num == curr_num - 1) or (curr_num == 1 and cand_num == 12)
                        
                        if is_up:
                            mix_label = f"{Fore.CYAN}[ENERGY]{Style.RESET_ALL}"
                            mix_desc = "Tonalidad adyacente - (Subir energía)"
                        elif is_down:
                            mix_label = f"{Fore.CYAN}[ENERGY]{Style.RESET_ALL}"
                            mix_desc = "Tonalidad adyacente - (Bajar energía/Relax)"

                # Indicar si es externo
                is_external = match in external_files
                ext_tag = ""
                if is_external:
                    # Obtener nombre de la carpeta contenedora
                    folder_name = os.path.basename(os.path.dirname(match['path']))
                    # Si la carpeta es muy larga, acortarla visualmente
                    if len(folder_name) > 15:
                        folder_name = folder_name[:12] + ".."
                    ext_tag = f"{Fore.MAGENTA}[{folder_name}]{Style.RESET_ALL} "
                
                print(f"  -> {mix_label} {ext_tag}{Fore.CYAN}[{match['camelot']}]{Style.RESET_ALL} {match['filename']} ({bpm_color}{match['bpm']} BPM{Style.RESET_ALL})")
                if mix_desc:
                    print(f"     {Fore.WHITE}   {mix_desc}{Style.RESET_ALL}")
    
    print("\n" + Fore.CYAN + "="*60 + Style.RESET_ALL)

def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        print("\n" + "!"*50)
        print("ERROR: ffmpeg no encontrado.")
        print("Para que este programa funcione correctamente (especialmente para ganancia y normalización),")
        print("necesitas instalar FFmpeg y agregarlo a tu PATH.")
        print("!"*50 + "\n")

def select_folder():
    """Abre un diálogo para seleccionar carpeta."""
    try:
        root = tk.Tk()
        root.withdraw() # Ocultar ventana principal
        # Asegurar que la ventana aparezca al frente
        root.attributes('-topmost', True)
        folder = filedialog.askdirectory(title="Selecciona la carpeta de música")
        root.destroy()
        return folder
    except Exception as e:
        print(f"Error al abrir selector de carpeta: {e}")
        return None

def select_multiple_folders():
    """Abre un diálogo para seleccionar múltiples carpetas."""
    try:
        root = tk.Tk()
        root.withdraw() # Ocultar ventana principal
        root.attributes('-topmost', True)
        
        folders = []
        while True:
            folder = filedialog.askdirectory(title="Selecciona una carpeta de música (Cancelar para terminar)")
            if not folder:
                break
            folders.append(folder)
            print(f"Carpeta agregada: {folder}")
            
            # Preguntar si quiere agregar más
            root.attributes('-topmost', False)
            response = input("¿Agregar otra carpeta? (s/n): ").lower().strip()
            if response not in ['s', 'si', 'sí', 'y', 'yes']:
                break
            root.attributes('-topmost', True)
        
        root.destroy()
        return folders
    except Exception as e:
        print(f"Error al abrir selector de carpetas: {e}")
        return []

def identify_missing_metadata(files):
    """Identifica archivos sin título o artista en metadatos y permite editarlos."""
    print(Fore.CYAN + "\n=== Archivos sin Metadatos (Título/Artista) ===" + Style.RESET_ALL)
    
    missing_files = []
    for f in files:
        # Check if title or artist is missing, empty, or 'Unknown'
        title_missing = not f.get('title') or f['title'] == 'Unknown'
        artist_missing = not f.get('artist') or f['artist'] == 'Unknown'
        
        if title_missing or artist_missing:
            missing_files.append(f)
            
    if not missing_files:
        print(Fore.GREEN + "¡Excelente! Todos los archivos tienen metadatos completos." + Style.RESET_ALL)
        return

    print(f"Se encontraron {len(missing_files)} archivos incompletos.")
    
    for i, f in enumerate(missing_files, 1):
        print(f"\n{i}. Archivo: {Fore.YELLOW}{f['filename']}{Style.RESET_ALL}")
        print(f"   Ruta: {f['path']}")
        print(f"   Título actual: {f.get('title', '---')}")
        print(f"   Artista actual: {f.get('artist', '---')}")
        
    print("\nOpciones:")
    print("1. Editar un archivo específico")
    print("2. Editar todos uno por uno")
    print("3. Autocompletar usando nombre de archivo (e.g. 'Artista - Titulo')")
    print("4. Volver al menú principal")
    
    choice = input("Opción: ").strip()
    
    if choice == '1':
        try:
            print("\nIngrese el número del archivo que desea editar (1, 2, 3...):")
            idx = int(input("Número: ")) - 1
            if 0 <= idx < len(missing_files):
                edit_metadata(missing_files[idx])
                # Actualizar caché para este archivo
                update_cache_for_files([missing_files[idx]])
            else:
                print("Número inválido.")
        except ValueError:
            print("Entrada inválida.")
            
    elif choice == '2':
        print("\nModo edición secuencial (Presiona Enter para mantener valor actual, o escribe el nuevo)")
        for f in missing_files:
            print(f"\nEditando: {Fore.YELLOW}{f['filename']}{Style.RESET_ALL}")
            edit_metadata(f)
            print("-" * 40)
        # Actualizar caché
        update_cache_for_files(missing_files)
            
    elif choice == '3':
        print("\nAutocompletar basado en nombre de archivo (e.g. Artista - Titulo.mp3)")
        confirm = input("¿Estás seguro? Esto sobrescribirá metadatos faltantes (s/n): ").lower()
        if confirm in ['s', 'si', 'y']:
            for f in missing_files:
                infer_metadata_from_filename(f)
            # Actualizar caché
            update_cache_for_files(missing_files)
            print(Fore.GREEN + "Proceso completado." + Style.RESET_ALL)
            
    elif choice == '4':
        return

def update_cache_for_files(files_to_update):
    """Actualiza el archivo de caché con los nuevos metadatos."""
    cache = load_cache()
    updated_count = 0
    # Convertir a dict para búsqueda rápida si son muchos
    paths_to_update = {f['path']: f for f in files_to_update}
    
    for path, data in cache.items():
        if path in paths_to_update:
            new_info = paths_to_update[path]
            # Actualizar info en caché con los datos modificados en memoria
            # Aseguramos que se guarden los campos clave
            if 'info' in data:
                data['info']['artist'] = new_info.get('artist', '')
                data['info']['title'] = new_info.get('title', '')
                updated_count += 1
    
    if updated_count > 0:
        save_cache(cache)
        print(f"{Fore.CYAN}[Caché actualizado con {updated_count} cambios]{Style.RESET_ALL}")

def infer_metadata_from_filename(file_info):
    """Intenta deducir Artista y Título desde el nombre de archivo."""
    filename = file_info['filename']
    name_without_ext = os.path.splitext(filename)[0]
    
    # Patrones comunes: 
    # 1. Artist - Title (Mix).mp3
    # 2. Artist_-_Title_(Mix).mp3
    # 3. Artist-Title.mp3
    
    # Reemplazar guiones bajos por espacios para facilitar análisis, PERO
    # manteniendo separadores clave.
    
    # Estrategia: Buscar el separador " - " o "_-_"
    
    artist = None
    title = None
    
    # Normalizar separadores:
    # 1. Reemplazar '–' (en dash) por '-'
    # 2. Reemplazar '_-_' por ' - '
    # 3. Reemplazar '_' por ' '
    
    clean_name = name_without_ext.replace('–', '-').replace('_-_', ' - ').replace('_', ' ')
    
    # Intento 1: Separador " - " (espacio guión espacio)
    if ' - ' in clean_name:
        parts = clean_name.split(' - ', 1)
        artist = parts[0].strip()
        title = parts[1].strip()
        
    # Intento 2: Separador "-" (solo guión)
    elif '-' in clean_name:
         parts = clean_name.split('-', 1)
         artist = parts[0].strip()
         title = parts[1].strip()
         
    # Intento 4: Regex para CamelCase "PaulParkerShotInTheNight" -> "Paul Parker" "Shot In The Night"
    # Este es muy específico y arriesgado, pero intentemos con la estructura básica de ArtistTitle
    if not artist and not title:
        # Buscamos un patrón donde haya mayúsculas que indiquen nuevas palabras
        # Asumimos que el artista suele ser las primeras 2 palabras o algo así? No, imposible saber.
        # Pero veamos el caso específico: "PaulParkerShotInTheNight"
        # Si separamos por mayúsculas: Paul Parker Shot In The Night
        # ¿Donde cortar?
        # A veces el usuario tiene archivos como "ArtistNameSongName"
        # Si no hay separador, no podemos adivinar 100% seguro.
        # PERO, si el usuario nos dio ejemplos específicos, podemos tratar de inferir.
        
        # Estrategia "Fuerza Bruta con Espacios":
        # Insertar espacio antes de cada mayúscula (excepto la primera)
        spaced_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', clean_name)
        # Ahora "Paul Parker Shot In The Night (12 Inch Version) (tancpol. net)"
        
        # Limpiar patrones basura PRIMERO en el nombre completo si no pudimos separar
        for pattern in [r'\(hydr0\.org\)', r'\(www\..*?\)', r'\(tancpol\.net\)', r'\(.*?tancpol\.net\)', r'\(.*?Amazing Italo Disco\)']:
            spaced_name = re.sub(pattern, '', spaced_name, flags=re.IGNORECASE).strip()
            
        # Intentar buscar palabras clave de separación comunes que quizás estaban pegadas
        # O simplemente, si no hay separador, reportar fallo pero mostrando lo que se logró limpiar
        
        # Caso específico: "The Boyd Brothers Keep It Coming" -> "The Boyd Brothers" - "Keep It Coming"
        # Imposible saber sin base de datos.
        
        # PERO, podemos intentar heurística: Si hay " - " después de insertar espacios?
        # No.
        
        # Si no detectamos separador, NO asignamos nada para evitar dañar datos, 
        # a menos que sea un caso muy obvio.
        pass

    if artist and title:
        # Limpieza adicional: eliminar patrones no deseados
        # Ejemplo: (Hydr0.org), (www.sitio.com), (tancpol.net), [algo]
        unwanted_patterns = [
            r'\(hydr0\.org\)', 
            r'\(www\..*?\)', 
            r'\(tancpol\.net\)',
            r'\[.*?\]',      # Eliminar contenido entre corchetes
            r'\(\d{4}.*?\)', # A veces años o info extraña entre paréntesis: (1984 Amazing Italo Disco)
            # CUIDADO: (1984 Amazing...) podría borrar algo útil, pero el usuario parece querer limpiar basura.
            # Vamos a ser específicos con lo que pidió:
            r'\(.*?tancpol\.net\)',
            r'\(.*?Amazing Italo Disco\)',
        ]
        
        for pattern in unwanted_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE).strip()
            artist = re.sub(pattern, '', artist, flags=re.IGNORECASE).strip()

        # Caso especial: CamelCase pegado "PaulParkerShotInTheNight"
        # Si el artista/titulo no tiene espacios y es largo, intentar separar CamelCase?
        # Riesgoso. Mejor nos enfocamos en limpiar lo que sí detectó.
        
        # Limpieza de "12InchVersion" pegado -> "12 Inch Version"
        # Esto requiere regex complejo. 
        
        # Corrección específica: Asegurar paréntesis en mezclas
        mix_patterns = ['Extended Mix', 'Original Mix', 'Club Mix', 'Radio Edit', 'Remix', '12 Inch Version', 'Maxi Version']
        for mix in mix_patterns:
             # Normalizar mix pegado: "12InchVersion" -> "12 Inch Version"
             collapsed = mix.replace(' ', '')
             if collapsed.lower() in title.lower() and mix.lower() not in title.lower():
                 # Reemplazar versión pegada por versión con espacios
                 # Usamos re.sub para case insensitive replace
                 title = re.sub(collapsed, mix, title, flags=re.IGNORECASE)

             # Poner paréntesis si faltan
             if title.lower().endswith(mix.lower()) and not title.endswith(')'):
                 title = re.sub(f"{mix}$", f"({mix})", title, flags=re.IGNORECASE)
                 
        # Limpiar espacios dobles
        title = re.sub(r'\s+', ' ', title).strip()
        artist = re.sub(r'\s+', ' ', artist).strip()
        
        print(f"Inquiriendo: {Fore.YELLOW}{filename}{Style.RESET_ALL}")
        print(f" -> Artista: {artist}")
        print(f" -> Título:  {title}")
        
        # Solo aplicar si falta el dato
        current_artist = file_info.get('artist')
        current_title = file_info.get('title')
        
        should_save = False
        
        if not current_artist or current_artist == 'Unknown':
            file_info['artist'] = artist
            should_save = True
            
        if not current_title or current_title == 'Unknown':
            file_info['title'] = title
            should_save = True
            
        if should_save:
            save_metadata(file_info)
    else:
        print(f"No se pudo deducir patrón para: {filename}")

def save_metadata(file_info):
    """Guarda los metadatos en memoria al archivo físico."""
    path = file_info['path']
    try:
        # Intentar cargar con mutagen.File (easy=True para abstracción)
        audio = mutagen.File(path, easy=True)
        
        # Caso específico para WAV que a veces mutagen.File no maneja bien con easy=True si no tiene tags previos
        if audio is None or path.lower().endswith('.wav'):
             if path.lower().endswith('.wav'):
                 # Manejo específico para WAV
                 # WAV puede usar ID3 o INFO chunks. ID3 es más común para players modernos.
                 # Intentamos usar ID3 sobre WAV
                 try:
                     from mutagen.wave import WAVE
                     from mutagen.id3 import ID3, TIT2, TPE1
                     
                     try:
                         audio = WAVE(path)
                     except Exception:
                         # Si falla abrir como WAVE, quizás está corrupto o es otro formato
                         print(Fore.RED + f"   No se pudo abrir como WAVE: {path}" + Style.RESET_ALL)
                         return

                     # Asegurar que tiene tags ID3
                     if audio.tags is None:
                         audio.add_tags()
                     
                     # WAVE usa tags ID3 estándar, no la interfaz "easy" directamente a veces
                     # Pero si usamos audio.tags (que es ID3), debemos usar frames ID3 (TIT2, TPE1)
                     # O podemos intentar envolver en EasyID3 si es soportado.
                     
                     # Opción segura: Usar frames estándar de ID3
                     if file_info.get('title'):
                         audio.tags.add(TIT2(encoding=3, text=file_info['title']))
                     if file_info.get('artist'):
                         audio.tags.add(TPE1(encoding=3, text=file_info['artist']))
                     
                     audio.save()
                     print(Fore.GREEN + "   [Guardado WAV]" + Style.RESET_ALL)
                     return
                     
                 except Exception as wav_err:
                     print(Fore.YELLOW + f"   Intento WAV nativo falló ({wav_err}), probando genérico..." + Style.RESET_ALL)
            
             # Reintentar lógica genérica para MP3 u otros si falló lo anterior
             if path.lower().endswith('.mp3'):
                from mutagen.mp3 import MP3
                from mutagen.easyid3 import EasyID3
                try:
                    audio = EasyID3(path)
                except mutagen.id3.ID3NoHeaderError:
                    audio = mutagen.File(path, easy=True)
                    if audio is None: return
                    audio.add_tags()
        
        # Guardado genérico (MP3, FLAC, M4A con soporte Easy)
        if audio is not None:
            if file_info.get('title'): audio['title'] = [file_info['title']]
            if file_info.get('artist'): audio['artist'] = [file_info['artist']]
            audio.save()
            print(Fore.GREEN + "   [Guardado]" + Style.RESET_ALL)
            
    except Exception as e:
        print(Fore.RED + f"   Error guardando: {e}" + Style.RESET_ALL)

def edit_metadata(file_info):
    """Edita los metadatos de un archivo usando mutagen."""
    path = file_info['path']
    
    print(f"Nombre de archivo: {file_info['filename']}")
    
    # Pre-calcular sugerencias basadas en nombre de archivo
    suggested_artist = ""
    suggested_title = ""
    
    clean_name = os.path.splitext(file_info['filename'])[0].replace('_-_', ' - ').replace('_', ' ')
    if ' - ' in clean_name:
        parts = clean_name.split(' - ', 1)
        suggested_artist = parts[0].strip()
        suggested_title = parts[1].strip()
    
    current_title = file_info.get('title', '')
    current_artist = file_info.get('artist', '')
    
    # Mostrar sugerencia si está vacío
    title_prompt = f"Nuevo Título [{current_title}]"
    if not current_title and suggested_title:
        title_prompt += f" (Sugerido: {suggested_title})"
        
    artist_prompt = f"Nuevo Artista [{current_artist}]"
    if not current_artist and suggested_artist:
        artist_prompt += f" (Sugerido: {suggested_artist})"

    new_title = input(f"{title_prompt}: ").strip()
    new_artist = input(f"{artist_prompt}: ").strip()
    
    # Si está vacío y hay sugerencia, usar sugerencia? No, mejor explícito.
    # Lógica: 
    # - Si usuario escribe algo -> usar eso
    # - Si usuario da Enter (vacío) -> mantener valor actual
    # - Si valor actual es vacío y usuario da Enter -> sigue vacío
    
    # OPCIONAL: Si el usuario escribe "auto", usar la sugerencia
    if new_title.lower() == 'auto' and suggested_title: new_title = suggested_title
    if new_artist.lower() == 'auto' and suggested_artist: new_artist = suggested_artist

    if not new_title: new_title = current_title
    if not new_artist: new_artist = current_artist
    
    # Actualizar dict en memoria
    file_info['title'] = new_title
    file_info['artist'] = new_artist
    
    save_metadata(file_info)

def main():
    check_ffmpeg()
    parser = argparse.ArgumentParser(description="Herramienta de análisis y gestión de MP3")
    parser.add_argument('--path', help='Directorio a escanear (opcional)')
    parser.add_argument('--no-gain', action='store_true', help='Saltar cálculo de ganancia (más rápido)')
    parser.add_argument('--csv', help='Nombre del archivo CSV para exportar (opcional)', default='reporte_audio.csv')
    args = parser.parse_args()

    target_paths = [args.path] if args.path else []
    
    # Si no se pasa argumento, preguntar
    if not target_paths:
        print("\n¿Deseas analizar una sola carpeta o múltiples carpetas?")
        print("1. Una sola carpeta")
        print("2. Múltiples carpetas")
        choice = input("Opción (1/2): ").strip()
        
        if choice == '2':
            print("\nSelecciona las carpetas que deseas analizar...")
            target_paths = select_multiple_folders()
        else:
            print("\nSelecciona la carpeta que contiene tus archivos MP3...")
            target_path = select_folder()
            target_paths = [target_path] if target_path else []
    
    if not target_paths:
        print("No se seleccionaron carpetas. Usando el directorio actual.")
        target_paths = ['.']
    else:
        print(f"Carpetas seleccionadas ({len(target_paths)}):")
        for i, path in enumerate(target_paths, 1):
            print(f"  {i}. {path}")

    # 1. Analizar
    files = scan_directory(target_paths, calculate_gain=not args.no_gain)
    if not files:
        print("No se encontraron archivos MP3.")
        return

    # 2. Reporte
    sorted_files = print_report(files)

    # 3. Exportar CSV
    export_csv(sorted_files, args.csv)

    # 4. Menú de acciones
    while True:
        print(Fore.CYAN + "\n¿Qué deseas hacer?" + Style.RESET_ALL)
        print("1. Normalizar ganancia (nivelar volumen)")
        print("2. Organizar archivos (Artista/Album)")
        print("3. Ver espectro de un archivo")
        print("4. Sugerir mezclas armónicas (Camelot)")
        print("5. Identificar archivos sin metadatos")
        print("6. Cambiar carpeta de análisis")
        print("7. Salir")
        
        choice = input("\nOpción: ")
        
        if choice == '1':
            try:
                target = float(input("Ingrese objetivo dBFS (default -20): ") or -20)
                normalize_gain(sorted_files, target)
            except ValueError:
                print("Valor inválido.")
        elif choice == '2':
            organize_files(sorted_files)
        elif choice == '3':
            print("\nArchivos disponibles:")
            for idx, f in enumerate(sorted_files):
                print(f"{idx+1}. {f['filename']}")
            try:
                idx = int(input("Seleccione número de archivo: ")) - 1
                if 0 <= idx < len(sorted_files):
                    show_spectrum(sorted_files[idx]['path'])
                else:
                    print("Índice inválido.")
            except ValueError:
                print("Entrada inválida.")
        elif choice == '4':
            suggest_mixes(sorted_files)
        elif choice == '5':
            identify_missing_metadata(sorted_files)
        elif choice == '6':
            # Recursión simple para reiniciar (no ideal pero funciona para script simple)
            main()
            return
        elif choice == '7':
            print(Fore.GREEN + "¡Hasta luego!" + Style.RESET_ALL)
            sys.exit(0)
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()

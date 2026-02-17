from flask import Flask, render_template, request, jsonify
import os
import audio_manager
import json
import hashlib
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# Inicializar Firebase
cred = credentials.Certificate("firebase-service-account.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Cargar caché al inicio
def get_library():
    cache = audio_manager.load_cache()
    library = []
    for path, data in cache.items():
        info = data.get('info', {})
        # Agregar tamaño para comparación rápida
        info['file_size'] = data.get('size', 0)
        # Asegurar que la información de carpeta esté incluida
        if 'folder' not in info and 'path' in info:
            info['folder'] = os.path.dirname(info['path'])
        if info.get('camelot') and info.get('camelot') != '---':
            library.append(info)
    return library

def calculate_file_hash(file_stream):
    """Calcula hash MD5 de un stream de archivo."""
    md5 = hashlib.md5()
    for chunk in iter(lambda: file_stream.read(4096), b""):
        md5.update(chunk)
    file_stream.seek(0) # Reset pointer
    return md5.hexdigest()

# Configuración
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Asegurar directorios
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def load_playlists():
    try:
        playlists_ref = db.collection('playlists')
        docs = playlists_ref.stream()
        playlists = {}
        for doc in docs:
            playlists[doc.id] = {**doc.to_dict(), 'id': doc.id}
        return playlists
    except Exception as e:
        print(f"Error cargando playlists desde Firebase: {e}")
        return {}

def save_playlists(playlists):
    # Esta función ya no es necesaria ya que Firebase maneja la persistencia automáticamente
    # Las operaciones se hacen directamente en Firestore
    pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/library')
def api_library():
    return jsonify(get_library())

def get_analyzed_folders():
    """Obtiene todas las carpetas únicas que han sido analizadas."""
    cache = audio_manager.load_cache()
    folders = set()
    
    for file_data in cache.values():
        info = file_data.get('info', {})
        if 'folder' in info:
            folders.add(info['folder'])
        elif 'path' in info:
            # Para compatibilidad con versiones anteriores
            folder = os.path.dirname(info['path'])
            folders.add(folder)
    
    return sorted(list(folders))

@app.route('/api/analyzed-folders', methods=['GET'])
def api_analyzed_folders():
    """Endpoint para obtener carpetas analizadas desde el caché."""
    folders = get_analyzed_folders()
    return jsonify(folders)

@app.route('/api/analyze', methods=['POST'])
def analyze_track():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    allowed_folders = request.form.get('allowed_folders')
    if allowed_folders:
        try:
            allowed_folders = json.loads(allowed_folders)
        except:
            allowed_folders = None

    if file:
        try:
            # Buscar en biblioteca por nombre (solo desde caché JSON)
            library = get_library()
            
            # Normalizar nombre para búsqueda
            upload_filename = file.filename
            existing_track = None
            
            for t in library:
                # Comparar solo por nombre (en modo JSON no tenemos acceso a tamaño de archivo)
                if t['filename'] == upload_filename:
                    existing_track = t
                    break
            
            if existing_track:
                print(f"Archivo encontrado en caché: {existing_track['filename']}")
                suggestions = find_matches(existing_track, allowed_folders)
                return jsonify({
                    'track': existing_track,
                    'suggestions': suggestions,
                    'source': 'library_cache'
                })

            # Si no está en caché, devolver error (no analizamos archivos nuevos)
            return jsonify({
                'error': 'Archivo no encontrado en la biblioteca. Solo se pueden analizar archivos existentes en el caché JSON.'
            }), 404
            
        except Exception as e:
            print(f"Error procesando archivo: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

@app.route('/api/mix-suggestions', methods=['POST'])
def mix_suggestions():
    data = request.json
    track_data = data.get('track')
    allowed_folders = data.get('allowed_folders')
    
    if not track_data:
        return jsonify({'error': 'No track provided'}), 400
        
    # Si viene de la librería, tiene path. Si es subido, tiene info completa.
    # Intentamos buscar en librería para asegurar datos frescos, 
    # pero si no tiene path (recién subido), usamos los datos tal cual.
    
    track_path = track_data.get('path')
    track = track_data # Por defecto usamos lo que nos mandan
    
    if track_path:
        # Buscar en librería para confirmar
        library = get_library()
        found = next((t for t in library if t['path'] == track_path), None)
        if found:
            track = found
    
    suggestions = find_matches(track, allowed_folders)
    return jsonify({
        'track': track,
        'suggestions': suggestions
    })

def find_matches(track, allowed_folders=None):
    library = get_library()
    
    # Filtrar por carpetas permitidas si se especifican (explicitly not None)
    if allowed_folders is not None:
        if len(allowed_folders) == 0:
            return [] # Si se pasó una lista vacía explícita, no hay resultados.

        # Normalizar paths para comparación robusta
        # En modo "solo JSON", no usamos abspath ni chequeos de sistema de archivos
        # Usamos comparaciones de strings simples normalizadas
        normalized_allowed = [f.lower().replace('\\', '/').strip('/') for f in allowed_folders]
        
        filtered_library = []
        for t in library:
            # Obtener path del track y normalizar
            track_path = t.get('path', '')
            if not track_path: continue
            
            track_path_norm = track_path.lower().replace('\\', '/').strip('/')
            
            # Verificar si el archivo está dentro de alguna carpeta permitida
            # Comprobación exacta: extraer la carpeta del track y comparar
            track_folder = os.path.dirname(track_path_norm) if '/' in track_path_norm else ''
            
            # Comparar con todas las carpetas permitidas
            if any(track_folder == allowed_folder for allowed_folder in normalized_allowed):
                filtered_library.append(t)
        library = filtered_library

    compatible_keys = audio_manager.get_compatible_keys(track['camelot'])
    matches = []
    
    track_path = track.get('path', '')
    # Obtener carpeta de forma segura sin depender del sistema de archivos
    track_folder = track.get('folder', '')
    if not track_folder and track_path:
        # Intento manual de extraer carpeta
        track_folder = os.path.dirname(track_path)
    
    for candidate in library:
        if candidate.get('path') == track_path: continue
        
        if candidate['camelot'] in compatible_keys:
            # Calcular prioridad
            try:
                priority = compatible_keys.index(candidate['camelot'])
            except ValueError:
                priority = 99
                
            # BPM diff
            bpm_diff = abs(candidate['bpm'] - track['bpm'])
            
            # PRIORIDAD 1: Misma carpeta (más importante que todo)
            candidate_folder = candidate.get('folder', os.path.dirname(candidate['path']) if candidate['path'] else '')
            same_folder_priority = 0 if candidate_folder == track_folder else 10
            
            # Clasificación y Prioridad Unificada
            mix_type = "HARMONIC"
            mix_desc = "Tonalidad adyacente"
            sort_priority = 2 # Default to Harmonic
            
            curr_num, curr_let = audio_manager.parse_camelot_code(track['camelot'])
            cand_num, cand_let = audio_manager.parse_camelot_code(candidate['camelot'])
            
            if curr_num is not None and cand_num is not None:
                if track['camelot'] == candidate['camelot']:
                    mix_type = "PERFECT"
                    mix_desc = "Misma tonalidad - (Mantiene la vibra)"
                    sort_priority = 1
                elif curr_num == cand_num and curr_let != cand_let:
                    mix_type = "HARMONIC"
                    sort_priority = 2
                    if curr_let == 'B' and cand_let == 'A':
                        mix_desc = "Cambio de Escala: De Alegre a Serio"
                    elif curr_let == 'A' and cand_let == 'B':
                        mix_desc = "Cambio de Escala: De Serio a Alegre"
                else:
                    is_up = (cand_num == curr_num + 1) or (curr_num == 12 and cand_num == 1)
                    is_down = (cand_num == curr_num - 1) or (curr_num == 1 and cand_num == 12)
                    
                    if is_up:
                        mix_type = "ENERGY"
                        mix_desc = "Subir energía (+1)"
                        sort_priority = 3
                    elif is_down:
                        mix_type = "ENERGY"
                        mix_desc = "Bajar energía (-1)"
                        sort_priority = 3

            matches.append({
                'track': candidate,
                'bpm_diff': bpm_diff,
                'priority': sort_priority, # Use unified priority
                'mix_type': mix_type,
                'mix_desc': mix_desc,
                'same_folder_priority': same_folder_priority  # Prioridad de misma carpeta
            })
            
    # Ordenar: Primero misma carpeta, luego compatibilidad armónica, luego diferencia BPM
    matches.sort(key=lambda x: (x['same_folder_priority'], x['priority'], x['bpm_diff']))
    return matches

# --- Playlist API ---

@app.route('/api/playlists', methods=['GET'])
def get_playlists():
    try:
        playlists_ref = db.collection('playlists').order_by('created_at', direction=firestore.Query.DESCENDING)
        docs = playlists_ref.stream()
        playlists = []
        for doc in docs:
            playlist_data = doc.to_dict()
            playlists.append({'id': doc.id, **playlist_data})
        return jsonify(playlists)
    except Exception as e:
        print(f"Error obteniendo playlists desde Firebase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists', methods=['POST'])
def create_playlist():
    data = request.json
    name = data.get('name', 'Nueva Playlist')
    
    try:
        playlist_ref = db.collection('playlists').document()
        playlist_data = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'tracks': []
        }
        playlist_ref.set(playlist_data)
        
        return jsonify({'id': playlist_ref.id, **playlist_data})
    except Exception as e:
        print(f"Error creando playlist en Firebase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/track-info', methods=['GET'])
def get_track_info():
    """Obtener información de una canción desde el cache."""
    path = request.args.get('path')
    filename = request.args.get('filename')
    
    if not path and not filename:
        return jsonify({'error': 'Path or filename parameter required'}), 400
    
    cache = audio_manager.load_cache()
    
    # Buscar por path (comparación de strings directa)
    if path:
        # En modo JSON, path es solo un identificador string
        if path in cache:
            return get_track_suggestions(cache[path])
    
    # Buscar por nombre de archivo
    if filename:
        # Buscar en todo el cache por coincidencia de nombre
        for cache_path, track_data in cache.items():
            track_info = track_data.get('info', {})
            cache_filename = track_info.get('filename', '')
            
            # Comparar con el nombre del archivo guardado en los metadatos
            if cache_filename == filename:
                return get_track_suggestions(track_data)
        
        # Si no encuentra coincidencia exacta, buscar coincidencias parciales
        for cache_path, track_data in cache.items():
            track_info = track_data.get('info', {})
            cache_filename = track_info.get('filename', '')
            
            if filename in cache_filename:
                return get_track_suggestions(track_data)
        
    return jsonify({'error': 'Track not found in cache'}), 404

def get_track_suggestions(track_data):
    """Obtener sugerencias para una pista."""
    track_info = track_data.get('info', {})
    suggestions = find_matches(track_info)
    
    return jsonify({
        'track': track_info,
        'suggestions': suggestions
    })

@app.route('/api/playlists/<playlist_id>', methods=['GET'])
def get_playlist(playlist_id):
    try:
        playlist_ref = db.collection('playlists').document(playlist_id)
        playlist_doc = playlist_ref.get()
        
        if playlist_doc.exists:
            playlist_data = playlist_doc.to_dict()
            return jsonify({'id': playlist_id, **playlist_data})
        else:
            return jsonify({'error': 'Playlist not found'}), 404
    except Exception as e:
        print(f"Error obteniendo playlist desde Firebase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>', methods=['DELETE'])
def delete_playlist(playlist_id):
    try:
        playlist_ref = db.collection('playlists').document(playlist_id)
        playlist_doc = playlist_ref.get()
        
        if playlist_doc.exists:
            playlist_ref.delete()
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Playlist not found'}), 404
    except Exception as e:
        print(f"Error eliminando playlist desde Firebase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>/tracks', methods=['POST'])
def add_track_to_playlist(playlist_id):
    try:
        playlist_ref = db.collection('playlists').document(playlist_id)
        playlist_doc = playlist_ref.get()
        
        if not playlist_doc.exists:
            return jsonify({'error': 'Playlist not found'}), 404
            
        data = request.json
        track = data.get('track', {})
        
        # Obtener tracks actuales y agregar el nuevo
        playlist_data = playlist_doc.to_dict()
        tracks = playlist_data.get('tracks', [])
        tracks.append(track)
        
        # Actualizar en Firebase
        playlist_ref.update({'tracks': tracks})
        
        return jsonify({'id': playlist_id, **playlist_data, 'tracks': tracks})
    except Exception as e:
        print(f"Error agregando track a playlist en Firebase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>/tracks', methods=['DELETE'])
def remove_track_from_playlist(playlist_id):
    try:
        playlist_ref = db.collection('playlists').document(playlist_id)
        playlist_doc = playlist_ref.get()
        
        if not playlist_doc.exists:
            return jsonify({'error': 'Playlist not found'}), 404
            
        data = request.json
        track_index = data.get('index')
        
        playlist_data = playlist_doc.to_dict()
        tracks = playlist_data.get('tracks', [])
        
        if track_index is not None and 0 <= track_index < len(tracks):
            tracks.pop(track_index)
            playlist_ref.update({'tracks': tracks})
            return jsonify({'id': playlist_id, **playlist_data, 'tracks': tracks})
            
        return jsonify({'error': 'Invalid track index'}), 400
    except Exception as e:
        print(f"Error eliminando track de playlist en Firebase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>/reorder', methods=['POST'])
def reorder_playlist(playlist_id):
    try:
        playlist_ref = db.collection('playlists').document(playlist_id)
        playlist_doc = playlist_ref.get()
        
        if not playlist_doc.exists:
            return jsonify({'error': 'Playlist not found'}), 404
        
        data = request.json
        new_tracks = data.get('tracks')
        
        if not isinstance(new_tracks, list):
            return jsonify({'error': 'Invalid tracks data'}), 400
            
        # Actualizar el orden de los tracks en Firebase
        playlist_ref.update({'tracks': new_tracks})
        
        playlist_data = playlist_doc.to_dict()
        return jsonify({'id': playlist_id, **playlist_data, 'tracks': new_tracks})
    except Exception as e:
        print(f"Error reordenando tracks en Firebase: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/playlists/<playlist_id>/rename', methods=['PUT'])
def rename_playlist(playlist_id):
    try:
        playlist_ref = db.collection('playlists').document(playlist_id)
        playlist_doc = playlist_ref.get()
        
        if not playlist_doc.exists:
            return jsonify({'error': 'Playlist not found'}), 404
        
        data = request.json
        new_name = data.get('name')
        
        if not new_name or not new_name.strip():
            return jsonify({'error': 'Invalid name'}), 400
            
        # Actualizar el nombre en Firebase
        playlist_ref.update({'name': new_name.strip()})
        
        playlist_data = playlist_doc.to_dict()
        return jsonify({'id': playlist_id, **playlist_data, 'name': new_name.strip()})
    except Exception as e:
        print(f"Error renombrando playlist en Firebase: {e}")
        return jsonify({'error': str(e)}), 500

# Preferencias de carpetas del usuario
@app.route('/api/preferences/folders', methods=['GET', 'POST'])
def manage_folder_preferences():
    try:
        if request.method == 'GET':
            # Obtener preferencias de carpetas
            prefs_ref = db.collection('preferences').document('folders')
            prefs_doc = prefs_ref.get()
            
            if prefs_doc.exists:
                prefs_data = prefs_doc.to_dict()
                return jsonify(prefs_data.get('enabled_folders', []))
            else:
                return jsonify([])
                
        elif request.method == 'POST':
            # Guardar preferencias de carpetas
            data = request.json
            enabled_folders = data.get('enabled_folders', [])
            
            prefs_ref = db.collection('preferences').document('folders')
            prefs_ref.set({
                'enabled_folders': enabled_folders,
                'updated_at': datetime.now().isoformat()
            })
            
            return jsonify({'success': True, 'enabled_folders': enabled_folders})
            
    except Exception as e:
        print(f"Error gestionando preferencias de carpetas: {e}")
        return jsonify({'error': str(e)}), 500

# Estado del escaneo
SCAN_STATUS = {
    'scanning': False,
    'current_folder': None,
    'current_file': None,
    'last_scan': None
}

def scan_worker(folders):
    global SCAN_STATUS
    SCAN_STATUS['scanning'] = True
    
    def progress_callback(filepath):
        SCAN_STATUS['current_file'] = os.path.basename(filepath)

    try:
        print("Iniciando escaneo en segundo plano...")
        for folder in folders:
            if not os.path.exists(folder): continue
            
            SCAN_STATUS['current_folder'] = folder
            print(f"Escaneando carpeta: {folder}")
            
            # audio_manager.scan_directory ya actualiza el caché y devuelve la lista de archivos
            audio_manager.scan_directory(folder, calculate_gain=True, callback=progress_callback)
            
        SCAN_STATUS['last_scan'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("Escaneo completado.")
    except Exception as e:
        print(f"Error en escaneo: {e}")
    finally:
        SCAN_STATUS['scanning'] = False
        SCAN_STATUS['current_folder'] = None
        SCAN_STATUS['current_file'] = None

@app.route('/api/folders', methods=['GET'])
def manage_folders():
    """Endpoint para obtener carpetas desde el caché JSON."""
    if request.method == 'GET':
        # Obtener carpetas únicas del audio_analysis_cache.json
        folders = get_analyzed_folders()
        return jsonify(folders)

@app.route('/api/scan', methods=['POST'])
def start_scan():
    global SCAN_STATUS
    if SCAN_STATUS['scanning']:
        return jsonify({'status': 'already_scanning', 'info': SCAN_STATUS})
        
    folders = load_library_folders()
    if not folders:
        return jsonify({'error': 'No folders configured'}), 400
        
    thread = threading.Thread(target=scan_worker, args=(folders,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/scan/status')
def scan_status():
    return jsonify(SCAN_STATUS)

if __name__ == '__main__':
    print("Iniciando servidor web...")
    print("Abre http://localhost:5000 en tu navegador")
    # host='0.0.0.0' permite acceso desde otros dispositivos en la red
    app.run(debug=True, port=5000, host='0.0.0.0')

#!/usr/bin/env python3
"""
CSV/TXT Organizer UI (Wide → Long) with sequential metadata dialogs, smart re-import, and filename inference.

FIXED: food_encounter now correctly labels ONLY when nose enters FIRST (0→1) while centroid stays 0
       (not when centroid is already on, or when both enter simultaneously)

What's new in this version
- Column order: `source_file` is now the FIRST column in the composite.
- The last columns are coordinates/flags in this exact order: `x`, `y`, `centroid_on_food`, `nose_on_food`, `food_encounter`.
- Supports both older wide format (x,y,flag) and NEW wide format from the masking pipeline that emits
  per-worm `*_x`, `*_y`, `*_centroid_on_food` (centroid vs mask) and `*__nose_on_food` (nose vs mask).
- Gracefully handles composite/long CSVs produced by this script: reorders columns and assigns
  `assay_num` for the current import batch (does not break if long files are added).
- Remembers last-entered metadata per session and pre-fills dialogs.
- `nose_on_food`: binary flag for when nose is on food
- `food_encounter`: marks "food" ONLY when nose enters first (0→1) while centroid stays 0 (true biological entry)
- NEW: Filename inference mode - automatically parse metadata from standardized filenames
  Format: PC1_5.28.2025_m_wt_3hr# (PC#_date_sex_strain_treatment#)

Output columns (final order):
  1) source_file
  2) assay_num
  3) track_num
  4) pc_number
  5) sex
  6) strain_genotype
  7) treatment
  8) time
  9...) (any extra columns carried through, if present)
  shape metrics and stim (if present): major_axis, minor_axis, aspect_ratio, area, perimeter, convexity, solidity, stim
  last-7) x, y, nose_x, nose_y, centroid_on_food, nose_on_food, food_encounter

Run: python merge_files.py
Requires: Python 3.9+, pandas
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "CSV/TXT Organizer - Worm Tracks (Wide → Long)"
DEFAULT_ANALYZE_SUBDIR = "analyze"

# ------------------------------
# Utility: delimiter inference
# ------------------------------
def _infer_delimiter(path: Path) -> str:
    delimiters = [',', '\t', ';', ' ']
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    sample = line
                    break
            else:
                return ','
    except Exception:
        return ','
    best, best_count = ',', -1
    for d in delimiters:
        c = sample.count(d)
        if c > best_count:
            best, best_count = d, c
    return best

# ------------------------------
# Filename parsing utilities
# ------------------------------
def normalize_date(date_str: str) -> str:
    """Normalize various date formats to a consistent format."""
    date_str = re.sub(r'[^\d\.\-\/]', '', date_str)
    date_str = re.sub(r'[\-\/]', '.', date_str)
    return date_str

def normalize_sex(sex_str: str) -> str:
    """Normalize sex field to standard lowercase format."""
    sex_lower = sex_str.lower().strip()
    sex_mapping = {
        'm': 'm', 'male': 'm',
        'f': 'f', 'female': 'f', 
        'h': 'h', 'hermaphrodite': 'h', 'herm': 'h'
    }
    return sex_mapping.get(sex_lower, sex_lower)

def parse_filename_metadata(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse metadata from filename with flexible ordering.
    Expected components: PC#, date, sex, strain/genotype, treatment
    
    Examples:
        PC8_5.2.2025_tph-1_h_3hr
        PC8_5.2.2025_h_tph-1_3hr
        PC1_5.28.2025_m_wt_3hr
    
    Returns dict with keys: pc_number, date, sex, strain_genotype, treatment
    Returns None if parsing fails.
    """
    basename = filename
    for ext in ['.csv', '.txt', '.CSV', '.TXT']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    
    parts = basename.split('_')
    
    if len(parts) < 4:
        return None
    
    try:
        # PC number (must be first)
        pc_part = parts[0].strip()
        if not re.match(r'^PC\d+$', pc_part, re.IGNORECASE):
            return None
        pc_number = pc_part.upper()
        
        # Date (must be second)
        date_part = parts[1].strip()
        if not re.search(r'\d', date_part):
            return None
        date = normalize_date(date_part)
        
        # Parse remaining parts flexibly
        remaining_parts = [p.strip() for p in parts[2:] if p.strip()]
        
        if len(remaining_parts) < 2:
            return None
        
        # Identify components
        sex_candidates = []
        strain_candidates = []
        treatment_candidates = []
        
        known_sex_values = {'m', 'f', 'h', 'male', 'female', 'hermaphrodite', 'herm'}
        treatment_patterns = [
            r'\d+hr$',
            r'\d+min$',
            r'^fed$',
        ]
        
        for i, part in enumerate(remaining_parts):
            part_lower = part.lower()
            
            if part_lower in known_sex_values:
                sex_candidates.append((i, part))
            elif any(re.search(pattern, part_lower) for pattern in treatment_patterns):
                treatment_candidates.append((i, part))
            else:
                strain_candidates.append((i, part))
        
        # Extract sex (required)
        if not sex_candidates:
            return None
        sex = normalize_sex(sex_candidates[0][1])
        
        # Extract treatment (default to 'fed')
        treatment = 'fed'
        treatment_idx = None
        if treatment_candidates:
            treatment = re.sub(r'#*$', '', treatment_candidates[0][1]).lower()
            treatment_idx = treatment_candidates[0][0]
        
        # Extract strain
        sex_idx = sex_candidates[0][0]
        used_indices = {sex_idx}
        if treatment_idx is not None:
            used_indices.add(treatment_idx)
        
        strain_parts = [remaining_parts[i] for i in range(len(remaining_parts)) 
                       if i not in used_indices]
        
        # Strip tokens that are not biological metadata.
        # Uses fuzzy matching so misspellings (e.g. 'BOERDEI', 'BORDRE') are
        # also caught. A token is stripped if:
        #   (a) its similarity to any known non-strain word is >= 0.75, OR
        #   (b) it is a standalone digit string (replicate number like _2, _3)
        # Known non-strain tokens: all variants of border/noborder and 'final'.
        _NON_STRAIN_WORDS = ['border', 'borders', 'noborder', 'noborders', 'final']

        def _is_non_strain(token: str) -> bool:
            t = token.lower()
            # Exact digit check first
            if re.fullmatch(r'\d+', t):
                return True
            # Strip trailing digits before fuzzy check (handles border2, final3 etc.)
            t_base = re.sub(r'\d+$', '', t)
            if not t_base:
                return True
            for word in _NON_STRAIN_WORDS:
                ratio = SequenceMatcher(None, t_base, word).ratio()
                if ratio >= 0.75:
                    return True
            return False

        strain_parts = [p for p in strain_parts if not _is_non_strain(p)]
        
        if not strain_parts:
            return None
        
        strain_genotype = '-'.join(strain_parts).lower()
        
        return {
            'pc_number': pc_number,
            'date': date,
            'sex': sex,
            'strain_genotype': strain_genotype,
            'treatment': treatment
        }
        
    except Exception:
        return None

def validate_parsed_metadata(metadata: Dict[str, str]) -> Tuple[bool, str]:
    """Validate parsed metadata. Returns (is_valid, error_message)."""
    if not metadata['pc_number']:
        return False, "PC number is required"
    
    if not re.match(r'^PC\d+$', metadata['pc_number'], re.IGNORECASE):
        return False, "PC number must be in format PC# (e.g., PC1)"
    
    valid_sex = {'m', 'f', 'h', 'male', 'female', 'hermaphrodite'}
    if metadata['sex'].lower() not in valid_sex:
        return False, f"Sex must be one of: {', '.join(valid_sex)}"
    
    return True, ""

# ------------------------------
# Robust file reading
# ------------------------------
def read_table(path: Path) -> pd.DataFrame:
    delim = _infer_delimiter(path)
    for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            df = pd.read_csv(path, sep=delim, engine='python', encoding=enc)
            if df.shape[1] == 1 and delim != '\t':  # try tab if single col
                df = pd.read_csv(path, sep='\t', engine='python', encoding=enc)
            return df
        except Exception:
            continue
    # last resort: whitespace
    try:
        return pd.read_csv(path, sep=r"\s+", engine='python')
    except Exception as e:
        raise RuntimeError(f"Failed to read {path.name}: {e}")

# ------------------------------
# Helpers to detect long format and normalize columns
# ------------------------------
LONG_REQUIRED = {'time'}
LONG_ANY_COORD = {'x', 'y'}
LONG_ANY_FLAGS = {'nose_on_food', 'centroid_on_food', 'food_encounter'}

def is_long_format(df: pd.DataFrame) -> bool:
    cols = {c.strip().lower() for c in df.columns}
    # Minimal long-format check: must have time and at least one of x/y OR one of the flags
    return ('time' in cols) and (len(LONG_ANY_COORD & cols) > 0 or len(LONG_ANY_FLAGS & cols) > 0)

def enforce_column_order(df: pd.DataFrame) -> pd.DataFrame:
    # Desired final order
    first = ['source_file', 'assay_num', 'track_num', 'pc_number', 'sex', 'strain_genotype', 'treatment', 'time']
    # Shape metrics and stim come before coordinates
    shape_and_stim = ['major_axis', 'minor_axis', 'aspect_ratio', 'area', 'perimeter', 'convexity', 'solidity', 'stim']
    # Coordinates and nose, then flags
    last = ['x', 'y', 'nose_x', 'nose_y', 'centroid_on_food', 'nose_on_food', 'food_encounter']
    existing = list(df.columns)

    out = []
    # Add first columns
    for c in first:
        if c in df.columns:
            out.append(c)

    # Middle: anything not in first/shape_and_stim/last
    for c in existing:
        if c not in out and c not in shape_and_stim and c not in last:
            out.append(c)

    # Add shape metrics and stim if present
    for c in shape_and_stim:
        if c in df.columns:
            out.append(c)

    # End: coords/nose/flags
    for c in last:
        if c in df.columns:
            out.append(c)

    return df[out]

# ------------------------------
# FIXED: Function to create food_encounter column
# Only labels when nose enters BEFORE or WITH centroid (not after)
# ------------------------------
def create_food_encounter_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates food_encounter column that marks 'food' only when the animal
    ENTERS the food with nose FIRST (centroid follows after).
    
    Valid food encounter - ONLY ONE CASE:
    - Nose goes 0→1 while centroid STAYS 0 (nose enters first, body follows later)
    
    Invalid (NOT labeled):
    - Centroid already 1 when nose goes 0→1 (animal already on food, tracking recovery)
    - Nose and centroid both go 0→1 simultaneously (both already on food)
    - Nose already 1 (nose was already touching food)
    
    The ONLY valid entry is: nose_before=0, nose_now=1, centroid_before=0, centroid_now=0
    This is the biological reality - the nose touches first, then the body moves onto food.
    """
    if 'nose_on_food' not in df.columns:
        df['food_encounter'] = ''
        return df
    
    df = df.copy()
    df['food_encounter'] = ''
    
    # Ensure centroid_on_food exists (if not, assume it follows nose)
    if 'centroid_on_food' not in df.columns:
        df['centroid_on_food'] = df['nose_on_food'].copy()
    
    # Group by track to handle each worm separately
    if 'track_num' in df.columns:
        for track_id in df['track_num'].unique():
            mask = df['track_num'] == track_id
            track_indices = df.loc[mask].index.tolist()
            
            # Get data as arrays for easier indexing
            nose_data = df.loc[mask, 'nose_on_food'].values
            centroid_data = df.loc[mask, 'centroid_on_food'].values
            
            # Find the first nose 0->1 transition, but only accept it if
            # centroid has NEVER been 1 at any prior frame. If centroid was
            # ever 1 before the transition, the animal had prior food contact
            # with unknown entry history (e.g. started on food, crossed into
            # off-food, then re-crossed back) and should be excluded entirely.
            centroid_ever_on = False
            for i in range(1, len(nose_data)):
                if centroid_data[i-1] == 1:
                    centroid_ever_on = True
                if centroid_ever_on:
                    continue  # skip all remaining transitions for this track
                if nose_data[i-1] == 0 and nose_data[i] == 1:
                    # centroid has never been 1 up to this point, valid encounter
                    original_idx = track_indices[i]
                    df.loc[original_idx, 'food_encounter'] = 'food'
                    break
    else:
        # If no track_num, treat entire dataset as one track
        nose_data = df['nose_on_food'].values
        centroid_data = df['centroid_on_food'].values if 'centroid_on_food' in df.columns else nose_data
        
        # Skip if starts on food
        if not (nose_data[0] == 1 or centroid_data[0] == 1):
            for i in range(1, len(nose_data)):
                if nose_data[i-1] == 0 and nose_data[i] == 1:
                    if centroid_data[i-1] == 0 and centroid_data[i] == 0:
                        df.iloc[i, df.columns.get_loc('food_encounter')] = 'food'
                        break
    
    return df

# ------------------------------
# Column detection / parsing for wide format
# ------------------------------
_TIME_CANDIDATES = ['time', 'frame', 't', 'frames']

def detect_time_column(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    for name in _TIME_CANDIDATES:
        for c in cols:
            if c.strip().lower() == name:
                return c
    return cols[0]

_CENTROID_FLAG_PAT = re.compile(r'centroid[_ ]?on[_ ]?food', re.I)
_X_PAT = re.compile(r'(?:^|[^a-z])(x|xpos|x_coord|xcoordinate|xposition)\b', re.I)
_Y_PAT = re.compile(r'(?:^|[^a-z])(y|ypos|y_coord|ycoordinate|yposition)\b', re.I)
_FLAG_PAT = re.compile(r'(?:flag|food|enc|onfood|nose_on_food|on_food|flag\d*)', re.I)

# Shape metrics patterns (for optogenetics experiments)
_MAJOR_AXIS_PAT = re.compile(r'major[_ ]?axis', re.I)
_MINOR_AXIS_PAT = re.compile(r'minor[_ ]?axis', re.I)
_ASPECT_RATIO_PAT = re.compile(r'aspect[_ ]?ratio', re.I)
_AREA_PAT = re.compile(r'(?:^|_)area(?:$|_)', re.I)
_PERIMETER_PAT = re.compile(r'perimeter', re.I)
_CONVEXITY_PAT = re.compile(r'convexity', re.I)
_SOLIDITY_PAT = re.compile(r'solidity', re.I)
_STIM_PAT = re.compile(r'(?:^|_)stim(?:$|_)', re.I)

# Nose coordinate patterns
_NOSE_X_PAT = re.compile(r'nose[_ ]?x', re.I)
_NOSE_Y_PAT = re.compile(r'nose[_ ]?y', re.I)

# List of shape metrics for easy iteration
SHAPE_METRICS = ['major_axis', 'minor_axis', 'aspect_ratio', 'area', 'perimeter', 'convexity', 'solidity']


def extract_id_token(colname: str) -> Optional[str]:
    """
    Extract worm/track ID from column name.
    Handles formats like:
    - worm_1_x, worm_1_major_axis -> '1'
    - worm1_x, worm1x -> '1'
    - x1, y1 -> '1'
    """
    # First try: worm_N_something pattern (most common in new tracker output)
    m = re.search(r'worm[_\s]*(\d+)[_\s]', colname, re.I)
    if m:
        return m.group(1)
    
    # Second try: digits at end (x1, y1, flag1)
    m = re.search(r'(\d+)\s*$', colname)
    if m:
        return m.group(1)
    
    # Third try: split by underscore/space and look for digit parts
    parts = re.split(r'[_\s]+', colname.strip())
    for part in parts:
        if part.isdigit():
            return part
    
    return None

def classify_col(colname: str) -> str:
    name = colname.strip().lower()
    if _CENTROID_FLAG_PAT.search(name):
        return 'centroid'
    # Check nose before general x/y to avoid misclassification
    if _NOSE_X_PAT.search(name): return 'nose_x'
    if _NOSE_Y_PAT.search(name): return 'nose_y'
    if _X_PAT.search(name): return 'x'
    if _Y_PAT.search(name): return 'y'
    if _FLAG_PAT.search(name): return 'flag'
    # Shape metrics
    if _MAJOR_AXIS_PAT.search(name): return 'major_axis'
    if _MINOR_AXIS_PAT.search(name): return 'minor_axis'
    if _ASPECT_RATIO_PAT.search(name): return 'aspect_ratio'
    if _AREA_PAT.search(name): return 'area'
    if _PERIMETER_PAT.search(name): return 'perimeter'
    if _CONVEXITY_PAT.search(name): return 'convexity'
    if _SOLIDITY_PAT.search(name): return 'solidity'
    # Stim marker
    if _STIM_PAT.search(name): return 'stim'
    # Fallback patterns
    if name.endswith('x'): return 'x'
    if name.endswith('y'): return 'y'
    if 'flag' in name or 'food' in name or 'enc' in name: return 'flag'
    return 'other'

def find_worm_sets(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping: worm_id -> {'x': col, 'y': col, 'nose_x': col?, 'nose_y': col?,
                                   'flag': col?, 'centroid': col?, 
                                   'major_axis': col?, 'minor_axis': col?, ..., 'stim': col?}
    """
    cols = [str(c) for c in df.columns]
    buckets: Dict[str, Dict[str, str]] = {}
    for c in cols:
        ctype = classify_col(c)
        if ctype == 'other':
            continue
        tok = extract_id_token(c) or c
        if tok not in buckets:
            buckets[tok] = {}
        buckets[tok][ctype] = c
    sets_ = {wid: sub for wid, sub in buckets.items() if 'x' in sub and 'y' in sub}
    return sets_

def normalize_flag_series(series: pd.Series) -> pd.Series:
    def to01(v):
        if pd.isna(v): return 0
        s = str(v).strip().lower()
        if s in ('1', 'true', 't', 'yes', 'y', '*', 'food', 'on', 'enc', 'encounter'):
            return 1
        try:
            return 1 if float(s) != 0.0 else 0
        except Exception:
            return 0
    return series.map(to01)

def parse_wide_file_to_long(path: Path,
                            assay_num: int,
                            pc_number: str,
                            sex: str,
                            strain_genotype: str,
                            treatment: str) -> pd.DataFrame:
    df = read_table(path)
    if df.empty or df.shape[1] < 2:
        raise RuntimeError(f"{path.name}: not enough columns to parse.")

    # If the file is already long format, just standardize/order columns and override assay_num
    if is_long_format(df):
        # Handle legacy food_encounter column (rename to nose_on_food if needed)
        if 'food_encounter' in df.columns and 'nose_on_food' not in df.columns:
            df = df.rename(columns={'food_encounter': 'nose_on_food'})
        
        # Ensure required columns exist / fill if missing
        for col in ['source_file', 'track_num', 'pc_number', 'sex', 'strain_genotype', 'treatment']:
            if col not in df.columns:
                if col == 'source_file':
                    df[col] = path.name
                elif col == 'track_num':
                    df[col] = 1
                else:
                    df[col] = ''
        
        # Ensure nose_on_food exists
        if 'nose_on_food' not in df.columns:
            df['nose_on_food'] = 0
            
        df['assay_num'] = assay_num
        
        # Create the new food_encounter column with corrected logic
        df = create_food_encounter_column(df)
        
        # Order columns and return
        df = enforce_column_order(df)
        return df

    # Otherwise, wide format parsing
    time_col = detect_time_column(df)
    time_series = df[time_col]
    sets_ = find_worm_sets(df)

    if not sets_:
        # attempt fallback: repeating groups
        other_cols = [c for c in df.columns if c != time_col]
        if len(other_cols) >= 2:
            group_size = 4 if len(other_cols) % 4 == 0 else (3 if len(other_cols) % 3 == 0 else None)
            if group_size is None:
                raise RuntimeError(f"{path.name}: could not detect worm column groups.")
            sets_ = {}
            n = len(other_cols) // group_size
            for i in range(n):
                grp = other_cols[group_size*i:group_size*i+group_size]
                entry = {'x': grp[0], 'y': grp[1]}
                if group_size == 4:
                    entry['centroid'] = grp[2]
                    entry['flag'] = grp[3]
                elif group_size == 3:
                    entry['flag'] = grp[2]
                sets_[str(i+1)] = entry
        else:
            raise RuntimeError(f"{path.name}: could not detect worm columns.")

    rows: List[pd.DataFrame] = []
    track_counter = 0

    def worm_sort_key(k: str):
        try:
            return int(re.sub(r'\D+', '', k) or '0')
        except Exception:
            return 0

    for worm_id in sorted(sets_.keys(), key=worm_sort_key):
        cols = sets_[worm_id]
        xcol, ycol = cols.get('x'), cols.get('y')
        flagcol = cols.get('flag')
        centcol = cols.get('centroid')
        nose_xcol = cols.get('nose_x')
        nose_ycol = cols.get('nose_y')

        if xcol is None or ycol is None:
            continue

        track_counter += 1
        x = pd.to_numeric(df[xcol], errors='coerce')
        y = pd.to_numeric(df[ycol], errors='coerce')

        if centcol and centcol in df.columns:
            centroid_flag = normalize_flag_series(df[centcol])
        else:
            centroid_flag = pd.Series(0, index=df.index)

        if flagcol and flagcol in df.columns:
            nose_flag = normalize_flag_series(df[flagcol])
        else:
            nose_flag = pd.Series(0, index=df.index)

        # Build base dataframe
        part = pd.DataFrame({
            'source_file': path.name,
            'assay_num': assay_num,
            'track_num': track_counter,
            'pc_number': pc_number,
            'sex': sex,
            'strain_genotype': strain_genotype,
            'treatment': treatment,
            'time': pd.to_numeric(time_series, errors='coerce'),
            'x': x,
            'y': y,
            'centroid_on_food': centroid_flag,
            'nose_on_food': nose_flag,
        })
        
        # Add nose coordinates if present
        if nose_xcol and nose_xcol in df.columns:
            part['nose_x'] = pd.to_numeric(df[nose_xcol], errors='coerce')
        if nose_ycol and nose_ycol in df.columns:
            part['nose_y'] = pd.to_numeric(df[nose_ycol], errors='coerce')
        
        # Add shape metrics if present (optogenetics experiments)
        for metric in SHAPE_METRICS:
            metric_col = cols.get(metric)
            if metric_col and metric_col in df.columns:
                part[metric] = pd.to_numeric(df[metric_col], errors='coerce')
        
        # Add stim column if present
        stim_col = cols.get('stim')
        if stim_col and stim_col in df.columns:
            part['stim'] = normalize_flag_series(df[stim_col])
        
        # Drop rows with NaN time
        part = part[~part['time'].isna()].reset_index(drop=True)
        
        # Create food_encounter column with corrected logic
        part = create_food_encounter_column(part)
        
        # Enforce final order
        part = enforce_column_order(part)
        rows.append(part)

    if not rows:
        raise RuntimeError(f"{path.name}: no valid worm x/y columns found.")
    return pd.concat(rows, ignore_index=True)

# ------------------------------
# UI components
# ------------------------------
class InputModeDialog(tk.Toplevel):
    """Dialog to choose between manual input and filename inference."""
    def __init__(self, master):
        super().__init__(master)
        self.title("Select Input Mode")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=15)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Choose how to enter experimental metadata:", 
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, columnspan=2, pady=(0,15))

        # Manual mode
        manual_frame = ttk.LabelFrame(frm, text="Manual Input", padding=10)
        manual_frame.grid(row=1, column=0, padx=(0,10), pady=5, sticky="nsew")
        
        ttk.Label(manual_frame, text="Enter metadata for each file\nthrough dialog boxes").grid(row=0, column=0, pady=5)
        ttk.Button(manual_frame, text="Use Manual Input", 
                   command=lambda: self._set_result('manual')).grid(row=1, column=0, pady=5)

        # Filename inference mode
        inference_frame = ttk.LabelFrame(frm, text="Filename Inference", padding=10)
        inference_frame.grid(row=1, column=1, padx=(10,0), pady=5, sticky="nsew")
        
        ttk.Label(inference_frame, text="Parse metadata from filenames\n\nExpected format:").grid(row=0, column=0, pady=2)
        ttk.Label(inference_frame, text="PC1_5.28.2025_m_wt_3hr#", 
                  font=('TkDefaultFont', 9, 'bold'), foreground='blue').grid(row=1, column=0, pady=2)
        ttk.Label(inference_frame, text="(PC#_date_sex_strain_treatment#)", 
                  font=('TkDefaultFont', 8), foreground='gray').grid(row=2, column=0, pady=2)
        ttk.Button(inference_frame, text="Use Filename Inference", 
                   command=lambda: self._set_result('inference')).grid(row=3, column=0, pady=5)

        # Cancel button
        ttk.Button(frm, text="Cancel", command=self._cancel).grid(row=2, column=0, columnspan=2, pady=(15,0))

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_visibility()
        self.focus_set()

    def _set_result(self, mode):
        self.result = mode
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

class FilenamePreviewDialog(tk.Toplevel):
    """Dialog to preview and confirm parsed metadata from filenames."""
    def __init__(self, master, parsed_data: List[Tuple[Path, Dict[str, str]]]):
        super().__init__(master)
        self.title("Preview Parsed Metadata")
        self.geometry("800x500")
        self.result = None
        self.parsed_data = parsed_data

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        ttk.Label(frm, text="Preview of parsed metadata (click rows to edit):", 
                  font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky="w", pady=(0,10))

        # Create treeview for preview
        columns = ('filename', 'pc_number', 'sex', 'strain_genotype', 'treatment', 'date')
        self.tree = ttk.Treeview(frm, columns=columns, show='headings', height=15)
        
        # Define headings
        self.tree.heading('filename', text='Filename')
        self.tree.heading('pc_number', text='PC Number')
        self.tree.heading('sex', text='Sex')
        self.tree.heading('strain_genotype', text='Strain/Genotype')
        self.tree.heading('treatment', text='Treatment')
        self.tree.heading('date', text='Date')

        # Set column widths
        self.tree.column('filename', width=150)
        self.tree.column('pc_number', width=80)
        self.tree.column('sex', width=60)
        self.tree.column('strain_genotype', width=120)
        self.tree.column('treatment', width=100)
        self.tree.column('date', width=100)

        self.tree.grid(row=1, column=0, sticky="nsew", pady=(0,10))
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(frm, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Populate tree
        for i, (path, metadata) in enumerate(parsed_data):
            self.tree.insert('', 'end', iid=i, values=(
                path.name,
                metadata['pc_number'],
                metadata['sex'],
                metadata['strain_genotype'],
                metadata['treatment'],
                metadata['date']
            ))

        # Bind double-click for editing
        self.tree.bind('<Double-1>', self._edit_item)

        # Buttons
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Edit Selected", command=self._edit_selected).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self._cancel).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Proceed", command=self._proceed).grid(row=0, column=2, padx=5)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    def _edit_item(self, event):
        selection = self.tree.selection()
        if selection:
            self._edit_selected()

    def _edit_selected(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a row to edit.")
            return
        
        item_id = int(selection[0])
        path, metadata = self.parsed_data[item_id]
        
        # Open edit dialog
        edit_dialog = FileMetaWizard(self, path.name, defaults=metadata)
        self.wait_window(edit_dialog)
        
        if edit_dialog.result:
            # Update the stored metadata and tree display
            self.parsed_data[item_id] = (path, edit_dialog.result)
            self.tree.item(selection[0], values=(
                path.name,
                edit_dialog.result['pc_number'],
                edit_dialog.result['sex'],
                edit_dialog.result['strain_genotype'],
                edit_dialog.result['treatment'],
                edit_dialog.result.get('date', metadata.get('date', ''))
            ))

    def _proceed(self):
        self.result = self.parsed_data
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

class FileMetaWizard(tk.Toplevel):
    """Modal dialog to collect metadata for a single file (prefills with last values)."""
    def __init__(self, master, fname: str, defaults: Optional[dict] = None):
        super().__init__(master)
        self.title(f"Metadata for file: {fname}")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        defaults = defaults or {}
        self.pc_var = tk.StringVar(value=defaults.get('pc_number', ''))
        self.sex_var = tk.StringVar(value=defaults.get('sex', ''))
        self.strain_var = tk.StringVar(value=defaults.get('strain_genotype', ''))
        self.treat_var = tk.StringVar(value=defaults.get('treatment', ''))

        def add_row(r, label, var, tip):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", padx=(0,8), pady=3)
            e = ttk.Entry(frm, textvariable=var, width=36)
            e.grid(row=r, column=1, sticky="ew", pady=3)
            ttk.Label(frm, text=tip, foreground="#666").grid(row=r, column=2, sticky="w", padx=(8,0))

        add_row(0, "PC #", self.pc_var, "(e.g., PC1)")
        add_row(1, "Sex", self.sex_var, "(e.g., M / H)")
        add_row(2, "Strain/Genotype", self.strain_var, "(e.g., N2 or tph-1)")
        add_row(3, "Treatment", self.treat_var, "(e.g., fed / 30min starved)")

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=3, pady=(10,0), sticky="e")
        ttk.Button(btns, text="Cancel", command=self._cancel).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="OK", command=self._ok).grid(row=0, column=1, padx=5)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_visibility()
        self.focus_set()

    def _ok(self):
        self.result = {
            'pc_number': self.pc_var.get().strip(),
            'sex': self.sex_var.get().strip(),
            'strain_genotype': self.strain_var.get().strip(),
            'treatment': self.treat_var.get().strip(),
        }
        self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1000x680")
        self.minsize(900, 600)

        # Remember last-entered metadata across files in a session
        self.last_meta = {
            'pc_number': '',
            'sex': '',
            'strain_genotype': '',
            'treatment': '',
        }

        style = ttk.Style(self)
        try:
            self.tk.call('tk', 'scaling', 1.25)
        except Exception:
            pass
        style.theme_use('clam')

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        self.new_tab = ttk.Frame(nb, padding=10)
        self.cont_tab = ttk.Frame(nb, padding=10)
        nb.add(self.new_tab, text="New Analysis")
        nb.add(self.cont_tab, text="Continue Analysis")
        self._build_new_tab()
        self._build_continue_tab()

    def _get_input_mode(self) -> Optional[str]:
        """Show dialog to choose input mode. Returns 'manual', 'inference', or None."""
        mode_dialog = InputModeDialog(self)
        self.wait_window(mode_dialog)
        return mode_dialog.result

    def _process_files_with_inference(self, files: List[Path]) -> Optional[List[Tuple[Path, Dict[str, str]]]]:
        """Process files using filename inference."""
        parsed_data = []
        failed_files = []

        for path in files:
            metadata = parse_filename_metadata(path.name)
            if metadata is None:
                failed_files.append(path.name)
                continue
            
            # Validate parsed metadata
            is_valid, error_msg = validate_parsed_metadata(metadata)
            if not is_valid:
                failed_files.append(f"{path.name}: {error_msg}")
                continue
                
            parsed_data.append((path, metadata))

        # Show any failed files
        if failed_files:
            failed_msg = "The following files could not be parsed:\n\n" + "\n".join(failed_files)
            failed_msg += "\n\nExpected format: PC#_date_sex_strain OR PC#_date_sex_strain_treatment"
            failed_msg += "\nTreatment defaults to 'fed' if not specified"
            failed_msg += "\nValid treatments: fed, 3hr, 6hr, 30min, etc."
            messagebox.showwarning("Parsing Failures", failed_msg)
            
            if not parsed_data:
                return None

        # Show preview dialog
        if parsed_data:
            preview_dialog = FilenamePreviewDialog(self, parsed_data)
            self.wait_window(preview_dialog)
            return preview_dialog.result
        
        return None

    def _process_files_manual(self, files: List[Path]) -> Optional[List[Tuple[Path, Dict[str, str]]]]:
        """Process files using manual input dialogs."""
        manual_data = []
        
        for path in files:
            md = FileMetaWizard(self, path.name, defaults=self.last_meta)
            self.wait_window(md)
            if md.result is None:
                # User cancelled
                response = messagebox.askyesnocancel(
                    "File Skipped", 
                    f"Skip {path.name} and continue with remaining files?\n\n"
                    "Yes = Skip this file\n"
                    "No = Abort entire process\n"
                    "Cancel = Go back to enter metadata"
                )
                if response is True:  # Skip
                    continue
                elif response is False:  # Abort
                    return None
                else:  # Go back
                    continue
            
            # Update defaults for next file
            self.last_meta = md.result.copy()
            manual_data.append((path, md.result))
            
        return manual_data if manual_data else None

    # -------- New Analysis Tab --------
    def _build_new_tab(self):
        frm = self.new_tab
        desc = ttk.Label(frm, text=(
            "Reads all .csv/.txt files from a directory (default: ./analyze),\n"
            "asks for metadata per file (manual or filename inference), and saves a composite long-format CSV.\n"
            "Supports new wide files with centroid_on_food and nose_on_food.\n"
            "FIXED: Creates food_encounter marking ONLY when nose enters FIRST (centroid stays 0).\n"
            "Filename format for inference: PC1_5.28.2025_m_wt_3hr# (PC#_date_sex_strain_treatment#)"
        ))
        desc.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,8))

        ttk.Label(frm, text="Data directory:").grid(row=1, column=0, sticky="e")
        self.new_dir_var = tk.StringVar(value=str((Path(sys.argv[0]).resolve().parent / DEFAULT_ANALYZE_SUBDIR)))
        ttk.Entry(frm, textvariable=self.new_dir_var, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Browse...", command=self._browse_new_dir).grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="Save composite as:").grid(row=2, column=0, sticky="e")
        self.new_save_var = tk.StringVar(value=str(Path.cwd() / "composite.csv"))
        ttk.Entry(frm, textvariable=self.new_save_var, width=60).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Choose...", command=self._choose_new_save).grid(row=2, column=2, sticky="w")

        ttk.Button(frm, text="Start Import", command=self._run_new_analysis).grid(row=3, column=1, pady=(10,0))

        self.new_log = tk.Text(frm, height=20, wrap="word")
        self.new_log.grid(row=4, column=0, columnspan=3, sticky="nsew", pady=(10,0))
        frm.rowconfigure(4, weight=1)
        frm.columnconfigure(1, weight=1)

    def _browse_new_dir(self):
        d = filedialog.askdirectory(title="Choose data directory")
        if d:
            self.new_dir_var.set(d)

    def _choose_new_save(self):
        f = filedialog.asksaveasfilename(title="Save composite as",
                                         defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if f:
            self.new_save_var.set(f)

    def _run_new_analysis(self):
        data_dir = Path(self.new_dir_var.get().strip())
        save_path = Path(self.new_save_var.get().strip())
        if not data_dir.exists() or not data_dir.is_dir():
            messagebox.showerror("Error", f"Data directory does not exist:\n{data_dir}")
            return

        files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in ('.csv', '.txt')])
        if not files:
            messagebox.showwarning("No files", f"No .csv/.txt files found in:\n{data_dir}")
            return

        # Get input mode
        input_mode = self._get_input_mode()
        if input_mode is None:
            return

        # Process files based on mode
        if input_mode == 'inference':
            file_data = self._process_files_with_inference(files)
        else:  # manual
            file_data = self._process_files_manual(files)

        if not file_data:
            self._log_new("Import cancelled or no valid files processed.")
            return

        # Process the files
        composite_rows: List[pd.DataFrame] = []
        assay_num = 1

        for path, metadata in file_data:
            try:
                part = parse_wide_file_to_long(
                    path=path,
                    assay_num=assay_num,
                    pc_number=metadata['pc_number'],
                    sex=metadata['sex'],
                    strain_genotype=metadata['strain_genotype'],
                    treatment=metadata['treatment'],
                )
                composite_rows.append(part)
                self._log_new(f"Parsed {path.name}: assay_num={assay_num}, rows={len(part)}")
                assay_num += 1
            except Exception as e:
                self._log_new(f"ERROR parsing {path.name}: {e}")

        if not composite_rows:
            messagebox.showwarning("Nothing saved", "No data imported.")
            return

        composite = pd.concat(composite_rows, ignore_index=True)
        try:
            composite = enforce_column_order(composite)
            composite.to_csv(save_path, index=False)
            self._log_new(f"Saved composite to: {save_path}")
            messagebox.showinfo("Done", f"Composite saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save composite:\n{e}")

    def _log_new(self, msg: str):
        self.new_log.insert("end", msg + "\n")
        self.new_log.see("end")
        self.new_log.update_idletasks()

    # -------- Continue Analysis Tab --------
    def _build_continue_tab(self):
        frm = self.cont_tab
        desc = ttk.Label(frm, text=(
            "Open an existing composite CSV, select a new data directory, and append new assays.\n"
            "Assay numbering continues from the last assay_num.\n"
            "FIXED: Recreates food_encounter - labels ONLY when nose enters first (centroid stays 0).\n"
            "Supports both manual input and filename inference for new files."
        ))
        desc.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,8))

        ttk.Label(frm, text="Existing composite:").grid(row=1, column=0, sticky="e")
        self.cont_comp_var = tk.StringVar(value=str(Path.cwd() / "composite.csv"))
        ttk.Entry(frm, textvariable=self.cont_comp_var, width=60).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Browse...", command=self._browse_composite).grid(row=1, column=2, sticky="w")

        ttk.Label(frm, text="New data directory:").grid(row=2, column=0, sticky="e")
        self.cont_data_dir_var = tk.StringVar(value=str((Path(sys.argv[0]).resolve().parent / DEFAULT_ANALYZE_SUBDIR)))
        ttk.Entry(frm, textvariable=self.cont_data_dir_var, width=60).grid(row=2, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Choose...", command=self._browse_cont_dir).grid(row=2, column=2, sticky="w")

        ttk.Label(frm, text="Save updated composite as:").grid(row=3, column=0, sticky="e")
        self.cont_save_var = tk.StringVar(value=str(Path.cwd() / "composite_updated.csv"))
        ttk.Entry(frm, textvariable=self.cont_save_var, width=60).grid(row=3, column=1, sticky="ew", padx=5)
        ttk.Button(frm, text="Choose...", command=self._choose_cont_save).grid(row=3, column=2, sticky="w")

        ttk.Button(frm, text="Append New Data", command=self._run_continue_analysis).grid(row=4, column=1, pady=(10,0))

        self.cont_log = tk.Text(frm, height=20, wrap="word")
        self.cont_log.grid(row=5, column=0, columnspan=3, sticky="nsew", pady=(10,0))
        frm.rowconfigure(5, weight=1)
        frm.columnconfigure(1, weight=1)

    def _browse_composite(self):
        f = filedialog.askopenfilename(title="Open existing composite CSV",
                                       filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if f:
            self.cont_comp_var.set(f)

    def _browse_cont_dir(self):
        d = filedialog.askdirectory(title="Choose new data directory")
        if d:
            self.cont_data_dir_var.set(d)

    def _choose_cont_save(self):
        f = filedialog.asksaveasfilename(title="Save updated composite as",
                                         defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if f:
            self.cont_save_var.set(f)

    def _run_continue_analysis(self):
        comp_path = Path(self.cont_comp_var.get().strip())
        if not comp_path.exists():
            messagebox.showerror("Error", f"Composite not found:\n{comp_path}")
            return
        try:
            composite = pd.read_csv(comp_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read composite:\n{e}")
            return

        # Handle legacy composite files
        if 'food_encounter' in composite.columns and 'nose_on_food' not in composite.columns:
            unique_vals = set(composite['food_encounter'].dropna().astype(str).str.lower())
            if unique_vals.issubset({'0', '1', '0.0', '1.0', 'true', 'false', 'nan'}):
                composite = composite.rename(columns={'food_encounter': 'nose_on_food'})
                composite['nose_on_food'] = normalize_flag_series(composite['nose_on_food'])
                self._log_cont("Renamed legacy food_encounter column to nose_on_food")
        
        # Ensure nose_on_food exists
        if 'nose_on_food' not in composite.columns:
            composite['nose_on_food'] = 0
        
        # Recreate food_encounter column with corrected logic
        composite['food_encounter'] = ''
        composite = create_food_encounter_column(composite)
        self._log_cont("Recreated food_encounter column with corrected logic (checks centroid state)")
        
        # Reorder columns
        composite = enforce_column_order(composite)

        if 'assay_num' not in composite.columns:
            messagebox.showerror("Error", "Composite missing 'assay_num' column.")
            return

        last_assay = int(pd.to_numeric(composite['assay_num'], errors='coerce').fillna(0).max())
        next_assay = last_assay + 1

        data_dir = Path(self.cont_data_dir_var.get().strip())
        if not data_dir.exists() or not data_dir.is_dir():
            messagebox.showerror("Error", f"New data directory does not exist:\n{data_dir}")
            return

        files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in ('.csv', '.txt')])
        if not files:
            messagebox.showwarning("No files", f"No .csv/.txt files found in:\n{data_dir}")
            return

        # Get input mode
        input_mode = self._get_input_mode()
        if input_mode is None:
            return

        # Process files
        if input_mode == 'inference':
            file_data = self._process_files_with_inference(files)
        else:
            file_data = self._process_files_manual(files)

        if not file_data:
            self._log_cont("Import cancelled or no valid files processed.")
            return

        new_rows: List[pd.DataFrame] = []
        assay_num = next_assay

        for path, metadata in file_data:
            try:
                part = parse_wide_file_to_long(
                    path=path,
                    assay_num=assay_num,
                    pc_number=metadata['pc_number'],
                    sex=metadata['sex'],
                    strain_genotype=metadata['strain_genotype'],
                    treatment=metadata['treatment'],
                )
                new_rows.append(part)
                self._log_cont(f"Parsed {path.name}: assay_num={assay_num}, rows={len(part)}")
                assay_num += 1
            except Exception as e:
                self._log_cont(f"ERROR parsing {path.name}: {e}")

        if not new_rows:
            messagebox.showwarning("Nothing appended", "No new data imported.")
            return

        updated = pd.concat([composite] + new_rows, ignore_index=True)
        try:
            updated = enforce_column_order(updated)
            save_path = Path(self.cont_save_var.get().strip())
            updated.to_csv(save_path, index=False)
            self._log_cont(f"Saved updated composite to: {save_path}")
            messagebox.showinfo("Done", f"Updated composite saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save updated composite:\n{e}")

    def _log_cont(self, msg: str):
        self.cont_log.insert("end", msg + "\n")
        self.cont_log.see("end")
        self.cont_log.update_idletasks()

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
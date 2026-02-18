# Fielddata.io FILEBOX Downloader

Tools for bulk-downloading files from fielddata.io FILEBOX.

## Quick start

```bash
# 1. Get a bearer token (see below)
# 2. Edit sections.json with your token and sections
cp sections_example.json sections.json
# Replace PASTE_BEARER_TOKEN_HERE with actual token

# 3. Download all sections
python3 download.py --config sections.json --dest ~/Downloads/output

# 4. Download a single section
python3 download.py --config sections.json --dest ~/Downloads/output --section 01_lignano_part2

# 5. Use more parallel workers (default: 4)
python3 download.py --config sections.json --dest ~/Downloads/output --workers 8
```

## How to get a bearer token

Tokens last ~24 hours. You need a fresh one each session.

### Method 1: Safari DevTools (recommended)

1. Log into https://www.fielddata.io in Safari
2. Navigate to any FILEBOX page
3. Open DevTools: **Develop > Show Web Inspector**
4. Go to **Network** tab
5. Click on any file to download it (or reload the page)
6. In the network requests, find any request to `www.fielddata.io/fileboxapi/...`
7. Click it, go to **Headers**, find `Authorization: Bearer eyJ...`
8. Copy everything after `Bearer ` — that's your token

### Method 2: Console interceptor

Paste this in Safari's console on the FILEBOX page, then reload:

```javascript
const _f = window.fetch;
window.fetch = async function(...a) {
  const h = a[1]?.headers;
  if (h) Object.entries(h).forEach(([k,v]) => {
    if (k.toLowerCase() === 'authorization')
      console.log('TOKEN:', v.replace('Bearer ',''));
  });
  return _f.apply(this, a);
};
console.log('Ready — reload the page (Cmd+R)');
```

## How to find project_id and dir_id

### From the URL

The FILEBOX URL format is:
```
https://www.fielddata.io/filebox/{project_id}/list/{dir_id}
```

### Using the listing tool

```bash
# List all directories in a project
python3 list_project.py --token "eyJ..." --project 4739

# List files in a specific directory
python3 list_project.py --token "eyJ..." --project 4739 --dir 10651
```

## API reference

All endpoints use `Authorization: Bearer {token}` header.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fileboxapi/directory/{projectId}` | GET | List all directories (flat, use parentId for tree) |
| `/fileboxapi/directory/{projectId}/{parentId}` | GET | List child directories of a parent |
| `/fileboxapi/file/{projectId}/directory/{dirId}` | GET | List files in a directory (returns `{files: [...]}`) |
| `/fileboxapi/file/{projectId}/{fileId}/download` | GET | Download a file (returns `{blob: "base64...", fileFormat: "..."}`) |

### File download format

The download endpoint does **not** return raw file content. It returns JSON:

```json
{
  "blob": "base64-encoded-file-content",
  "fileFormat": "application/json"
}
```

You must base64-decode the `blob` field to get the actual file content.
This applies to both `.json` and `.dat` files.

### File types

- `.json` files: UTF-8 JSON with equipment/sensor data (time series, events, parameters)
- `.dat` files: ISO-8859-1 text with CSV-like sensor data (comma-separated values, no line terminators)

## Auth details

- Auth provider: Auth0 (tenant: `fielddataio.eu.auth0.com`, issuer: `https://id.fielddata.io/`)
- Client ID: `1Ow627YBnykkYhwaANUJD7tUXGy51wEp`
- Token audience: `https://portal.fielddata.io/usermgt`
- Password grant is **disabled** — must use browser-based auth flow
- Tokens expire after ~24 hours

## Known project IDs (as of Feb 2026)

| Project ID | Name |
|------------|------|
| 4739 | 1529 Lignano Sabbiadoro |
| 2405 | 1508 Vicenza |
| 2440 | 1461 Invitalia |
| 3203 | Paris L18.3 OA20 (BG-45-V) |
| 4257 | 1454 Paris L18.3 OA20 (MC-86) |
| 3189 | 1511 Roma Acquedotto Marcio |
| 4600 | 1514 Catania |
| 5055 | 1502 MISP |

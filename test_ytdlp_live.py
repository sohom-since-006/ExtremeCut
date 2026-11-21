import yt_dlp
opts = {'quiet': True, 'no_warnings': True, 'noplaylist': True, 'format': 'b[height<=720]/bv*[height<=720]+ba/b'}
with yt_dlp.YoutubeDL(opts) as ydl:
    try:
        info = ydl.extract_info('https://www.youtube.com/@IShowSpeed/live', download=False)
        fmts = info.get('requested_formats') or [info]
        print('Number of requested formats:', len(fmts))
        for i, f in enumerate(fmts):
            fid = f.get('format_id')
            vc = f.get('vcodec')
            ac = f.get('acodec')
            print(f"Format {i}: id={fid}, vcodec={vc}, acodec={ac}")
            print("URL prefix:", f.get('url', '')[:80])
    except Exception as e:
        print('Error:', e)

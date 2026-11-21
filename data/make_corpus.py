"""Writes the made-up product manuals used as the RAG corpus (fictional brand NorthLight)."""
from pathlib import Path

MANUALS = {
"switcher": ("NorthLight NL-4 Production Switcher", {
 "Overview": "The NL-4 is a four mix-effects production switcher for live broadcast. Each mix-effects bank has eight keyers. The switcher accepts up to 96 inputs and drives 48 outputs. A single NL-4 frame supports formats up to 2160p at 59.94 frames per second.",
 "Transitions": "A transition moves the program output from one source to another. The NL-4 supports cut, mix, wipe and DVE transitions. The default mix rate is 30 frames. Press AUTO TRANS to run a transition at the stored rate, or move the fader bar for a manual transition.",
 "Keyers": "Keyers layer graphics or video over the background. The NL-4 offers luma keys, chroma keys and linear keys. The chroma keyer has automatic spill suppression. Up to eight keyers are available on every mix-effects bank.",
 "Memory": "Memories store the full state of a mix-effects bank. The NL-4 holds 100 memories per bank. Recall a memory from the control panel or with the memory recall command over the automation protocol.",
 "Redundancy": "The NL-4 frame has two hot-swappable power supplies. If one supply fails the frame keeps running on the other and raises a power alarm. The control processor can be fitted with an optional backup card that takes over within two frames.",
 "Troubleshooting": "If the program output freezes, check the reference signal first. The NL-4 must be locked to black burst or tri-level sync. A blinking REF light on the front panel means the frame has lost reference.",
}),
"camera": ("NorthLight CCU-7 Camera Control Unit", {
 "Overview": "The CCU-7 camera control unit links a studio camera to the control room over a single SMPTE fiber. It carries video, intercom, tally and control. One CCU-7 supports one camera head.",
 "Iris and Shading": "The shading operator adjusts iris, master black and gain from a remote control panel. Gain can be set from minus 6 decibels to plus 18 decibels in 3 decibel steps. Auto iris is disabled by default during live shows.",
 "Tally": "Tally lights tell the camera operator and talent which camera is on air. Red tally means program and green tally means preview. The CCU-7 receives tally from the production switcher over GPI or IP.",
 "Fiber Link": "The SMPTE fiber link can run up to 3 kilometres. The CCU-7 shows optical power on its front panel. If optical power falls below minus 20 dBm a fiber warning appears and the link should be cleaned.",
 "Intercom": "The CCU-7 carries two intercom channels to the camera operator. Channel one is production and channel two is engineering. Program audio can be sent to the operator headset as well.",
}),
"graphics": ("NorthLight Canvas Graphics Engine", {
 "Overview": "Canvas is a real-time graphics engine for on-air graphics such as lower thirds, scoreboards and full-screen data. Templates are built in the Canvas designer and played out by the Canvas server.",
 "Templates": "A template is a reusable graphic with text fields, images and animations. Data fields in a template can be filled by an operator or linked to a live data source. Templates are saved in the project library.",
 "Data Sources": "Canvas can read live data from CSV files, databases, REST APIs and sports scoring feeds. Data refreshes every second by default. A data source that stops updating turns amber in the data monitor.",
 "Playout": "Graphics are played out on two independent channels with fill and key outputs. Each channel can preview the next graphic before it goes to air. The take command moves the previewed graphic to air.",
 "Performance": "Canvas renders at the frame rate of the output format. Large textures should be kept under 4096 by 4096 pixels. If the render time goes above one frame, the performance meter turns red and frames may drop.",
}),
"replay": ("NorthLight Instant Replay Server", {
 "Overview": "The replay server records multiple camera feeds and plays them back for replays and highlights. A standard system records eight channels and plays back two channels at the same time.",
 "Recording": "Recording is continuous on all channels. The server keeps 24 hours of video on its internal storage. When storage is full the oldest material is overwritten unless it has been saved to a clip.",
 "Clips": "Clips are marked with an in point and an out point. Saved clips are protected from overwriting. Clips can be organised into playlists for highlight packages.",
 "Slow Motion": "Playback speed is controlled with the T-bar on the replay controller. Speeds from minus 100 percent to plus 200 percent are supported. High frame rate cameras give smoother slow motion at 3 times speed capture.",
 "Export": "Clips can be exported as files to network storage. Supported export formats are XAVC and ProRes. Export runs in the background and does not stop recording.",
}),
"router": ("NorthLight Matrix Video Router", {
 "Overview": "The Matrix router connects any input to any output. The largest frame is 288 inputs by 288 outputs. Routers in a facility can be controlled from hardware panels or software panels.",
 "Salvos": "A salvo is a group of routes taken together with one button. Salvos are used to set up a studio for a show in a single step. Up to 1000 salvos can be stored.",
 "Locks and Protects": "A lock stops anyone changing a destination. A protect stops other users changing it but lets the owner change it. Locks are used on transmission outputs during live shows.",
 "Control Protocol": "Third-party systems control the router over TCP using the router control protocol on port 52116. Each command routes one source to one destination and the router replies with the new state.",
}),
"gateway": ("NorthLight Remote Production Gateway", {
 "Overview": "The remote production gateway sends live video between a venue and a central control room over IP networks. It encodes up to four channels of video with low delay.",
 "Network": "The gateway sends video as SRT streams over UDP. It needs at least 20 megabits per second of upload bandwidth per HD channel. Packet loss recovery adds a buffer that is set to four times the round-trip time.",
 "Latency": "End-to-end delay is usually between 150 and 400 milliseconds depending on the network. Lower buffer values reduce delay but make the stream more sensitive to packet loss.",
 "Security": "Streams can be encrypted with AES-256 using a shared passphrase. Each gateway has its own administrator account, and the web interface should only be reached over HTTPS.",
 "Monitoring": "The gateway dashboard shows bitrate, round-trip time and packet loss for each stream. An alarm is raised if packet loss stays above 2 percent for more than ten seconds.",
}),
}

def main():
    out = Path(__file__).parent / "manuals"
    out.mkdir(exist_ok=True)
    for key, (title, sections) in MANUALS.items():
        lines = [f"# {title}", "", "Made-up sample manual for a demo. Not a real product.", ""]
        for name, text in sections.items():
            lines += [f"## {name}", "", text, ""]
        (out / f"{key}.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()

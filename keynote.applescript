-- Stack figures/keynote/build_*.png full-slide on one black 16:9 slide.
-- Usage: osascript keynote.applescript "$PWD"   (writes figures/lazuli_lensed_agn.key)
on run argv
	set repo to item 1 of argv
	set layers to paragraphs of (do shell script "ls " & quoted form of (repo & "/figures/keynote") & " | grep '^build_.*\\.png$' | sort")
	tell application "Keynote"
		activate
		set d to make new document with properties {document theme:theme "Basic Black", width:1920, height:1080}
		tell d
			set s to slide 1
			set base slide of s to slide layout "Blank"
			repeat with name_ in layers
				set f to POSIX file (repo & "/figures/keynote/" & name_)
				tell s to make new image with properties {file:f, position:{0, 0}, width:1920, height:1080}
			end repeat
			set out to count of images of s
		end tell
		save d in POSIX file (repo & "/figures/lazuli_lensed_agn.key")
		return out
	end tell
end run

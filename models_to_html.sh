mkdir models_html
for file in models/*; do
	base="${file##*/}"
	basename="${base%.*}"
	hn="models_html/$basename.html"
	cp template.html $hn
	sed -i "/<weights>/{
		r $file
		d
	}" "$hn"
	activation="${basename%%_*}"
	sed -i "s|<activation>|$activation|g" $hn
	lw="${basename#*_}"
	lw="${lw#*_}"
	lw="${lw#*_}"
	lw="${lw%%_*}"
	if [[ "$lw" == "32" ]]; then
		sed -i 's/<l1>/32/g' $hn
		sed -i 's/<l2>/16/g' $hn
	else
		sed -i 's/<l1>/256/g' $hn
		sed -i 's/<l2>/128/g' $hn
	fi
done
xquery version "3.0";

declare namespace tei="http://www.tei-c.org/ns/1.0";
declare namespace lido="http://www.lido-schema.org";

declare namespace output = "http://www.w3.org/2010/xslt-xquery-serialization"; 
declare option output:method "text";
declare option output:indent "no";


(: Define the collection to be processed :)
let $collection := "vif" 

(: Retrieve the file based on the collection selection. Note the switch-case construct :)
let $file :=
    switch ($collection)
        case "polos" return collection('xml/xml-polos')
        case "siba" return collection('xml/xml-siba')
        case "vif" return collection('xml/xml-vif')
        case "baci" return collection('xml/xml-baci')
        case "vismig" return collection('xml/xml-vismig')
        default return ()


(: Define CSV row header per file based on the collection :)
let $csv-row-headerPerFile := 
    switch ($collection)
        
        case "vif" return '"PID", "Creator", "Object title", "Type", "Format", "Licence/Copyright", "when", "notBefore", "notAfter", "OCM", "Internal keywords", "Image number", "Image 1", "Image 2", "Image 3", "Image 4", "Image 5", "Image 6", "Image 7", "Image 8", "Image 9", "Image 10", "Image 11", "Image 12", "Image 13", "Image 14", "Image 15", "Image 16", "Image 17", "Image 18", "Image 19"'
        
        case "baci" return '"PID", "Creator", "Object title", "Type", "Format", "Licence/Copyright", "when", "notBefore", "notAfter", "OCM", "Internal keywords", "Image number", "Image 1"'
        
        case "siba" return '"PID", "Creator", "Object title", "Type", "Format", "Licence/Copyright", "when", "notBefore", "notAfter", "OCM", "Internal keywords", "Image number", "Image 1"'
        
        case "vismig" return '"PID", "Creator", "Object title", "Type", "Format", "Licence/Copyright", "when", "notBefore", "notAfter", "OCM", "Internal keywords", "Image number", "Image 1", "Image 2", "Image 3", "Image 4", "Image 5", "Image 6", "Image 7", "Image 8", "Image 9", "Image 10", "Image 11", "Image 12", "Image 13", "Image 14", "Image 15", "Image 16", "Image 17", "Image 18"'
        
        case "polos" return '"PID", "Creator", "Object title", "Type", "Format", "Licence/Copyright", "when", "notBefore", "notAfter", "OCM", "Internal keywords", "Image number", "Image 1", "Image 2"'
        
    default return ''
        
 (: Create a CSV row for each document in the collection. Each case corresponds to a different collection :)       
let $csv-row :=
    switch ($collection)
        
        (: The following is done for each document in the collection "vase": :)
        case "vif" return 
            for $f in $file//TEI 
                let $pid := $f//idno[@type="PID"]
                let $creator := $f//ab[@subtype="creation"]//persName/ref/concat(.,
                            if(not(position()=last()))
                             then ' or '
                             else ()
                            )
                let $title := $f//titleStmt/title/translate(., '"', '')
                let $type := $f//ab[@type='lido:objectWorkType']
                let $format := $f//seg[@type='format']
                let $when := $f//body//ab[@type="lido:subject"]//date/@when/string()
                let $notBefore := $f//body//ab[@type="lido:subject"]//date/@notBefore/string()
                let $notAfter := $f//body//ab[@type="lido:subject"]//date/@notAfter/string()
                let $ocm := string-join(distinct-values($f//term[starts-with(@key, 'ocm')]/@key/substring-after(string(), 'ocm:')), '; ')
                let $i_keywords := ''
                let $imgNumber := count($f//graphic)
                let $img1 := if ($f//graphic[@xml:id='IMAGE.1']) then (concat(substring-after($pid, 'o:'), '-IMAGE.1.jpg')) else ''
                let $img2 := if ($f//graphic[@xml:id='IMAGE.2']) then (concat(substring-after($pid, 'o:'), '-IMAGE.2.jpg')) else ''
                let $img3 := if ($f//graphic[@xml:id='IMAGE.3']) then (concat(substring-after($pid, 'o:'), '-IMAGE.3.jpg')) else ''
                let $img4 := if ($f//graphic[@xml:id='IMAGE.4']) then (concat(substring-after($pid, 'o:'), '-IMAGE.4.jpg')) else ''
                let $img5 := if ($f//graphic[@xml:id='IMAGE.5']) then (concat(substring-after($pid, 'o:'), '-IMAGE.5.jpg')) else ''
                let $img6 := if ($f//graphic[@xml:id='IMAG-E.6']) then (concat(substring-after($pid, 'o:'), '-IMAGE.6.jpg')) else ''
                let $img7 := if ($f//graphic[@xml:id='IMAGE.7']) then (concat(substring-after($pid, 'o:'), '-IMAGE.7.jpg')) else ''
                let $img8 := if ($f//graphic[@xml:id='IMAGE.8']) then (concat(substring-after($pid, 'o:'), '-IMAGE.8.jpg')) else ''
                let $img9 := if ($f//graphic[@xml:id='IMAGE.9']) then (concat(substring-after($pid, 'o:'), '-IMAGE.9.jpg')) else ''
                let $img10 := if ($f//graphic[@xml:id='IMAGE.10']) then (concat(substring-after($pid, 'o:'), '-IMAGE.10.jpg')) else ''
                let $img11 := if ($f//graphic[@xml:id='IMAGE.11']) then (concat(substring-after($pid, 'o:'), '-IMAGE.11.jpg')) else ''
                let $img12 := if ($f//graphic[@xml:id='IMAGE.12']) then (concat(substring-after($pid, 'o:'), '-IMAGE.12.jpg')) else ''
                let $img13 := if ($f//graphic[@xml:id='IMAGE.13']) then (concat(substring-after($pid, 'o:'), '-IMAGE.13.jpg')) else ''
                let $img14 := if ($f//graphic[@xml:id='IMAGE.14']) then (concat(substring-after($pid, 'o:'), '-IMAGE.14.jpg')) else ''
                let $img15 := if ($f//graphic[@xml:id='IMAGE.15']) then (concat(substring-after($pid, 'o:'), '-IMAGE.15.jpg')) else ''
                let $img16 := if ($f//graphic[@xml:id='IMAGE.16']) then (concat(substring-after($pid, 'o:'), '-IMAGE.16.jpg')) else ''
                let $img17 := if ($f//graphic[@xml:id='IMAGE.17']) then (concat(substring-after($pid, 'o:'), '-IMAGE.17.jpg')) else ''
                let $img18 := if ($f//graphic[@xml:id='IMAGE.18']) then (concat(substring-after($pid, 'o:'), '-IMAGE.18.jpg')) else ''
                let $img19 := if ($f//graphic[@xml:id='IMAGE.19']) then (concat(substring-after($pid, 'o:'), '-IMAGE.19.jpg')) else ''
                order by $pid
                return ('&#10;"',$pid,'","',$creator,'","',$title,'","',$type,'","',$format,'","Creative Commons [CC BY-NC-ND 3.0]","',$when,'","',$notBefore,'","',$notAfter,'","',$ocm,'","',$i_keywords,'","',$imgNumber,'","',$img1,'","',$img2,'","',$img3,'","',$img4,'","',$img5,'","',$img6,'","',$img7,'","',$img8,'","',$img9,'","',$img10,'","',$img11,'","',$img12,'","',$img13,'","',$img14,'","',$img15,'","',$img16,'","',$img17,'","',$img18,'","',$img19,'"')
        
        (: The following is done for each document in the collection "baci": :)
        case "baci" return 
            for $f in $file//TEI 
                let $pid := $f//idno[@type="PID"]
                let $creator := $f//ab[@subtype="creation"]//persName/ref/concat(.,
                            if(not(position()=last()))
                             then ' or '
                             else ()
                            )
                let $title := $f//titleStmt/title/translate(., '"', '')
                let $type := $f//ab[@type='lido:objectWorkType']
                let $format := $f//seg[@type='format']
                let $copyright := $f//ab[@type="copyright"]
                let $when := $f//body//ab[@type="lido:subject"]//date/@when/string()
                let $notBefore := $f//body//ab[@type="lido:subject"]//date/@notBefore/string()
                let $notAfter := $f//body//ab[@type="lido:subject"]//date/@notAfter/string()
                let $ocm := string-join(distinct-values($f//term[starts-with(@key, 'ocm')]/@key/substring-after(string(), 'ocm:')), '; ')
                let $i_keywords := ''
                let $imgNumber := count($f//graphic)
                (:    es gibt nur IMAGE.1! :)
                let $img1 := if ($f//graphic[@xml:id='IMAGE.1']) then (concat(substring-after($pid, 'o:'), '-IMAGE.1.jpg')) else ''
                order by $pid
                return ('&#10;"',$pid,'","',$creator,'","',$title,'","',$type,'","',$format,'","',$copyright,'","',$when,'","',$notBefore,'","',$notAfter,'","',$ocm,'","',$i_keywords,'","',$imgNumber,'","',$img1,'"')
        
        (: The following is done for each document in the collection "siba": :)
        case "siba" return 
            for $f in $file//tei:TEI 
                let $pid := $f//tei:idno[@type="PID"]
                let $creator := $f//tei:ab[@subtype="creation"]//tei:persName/tei:ref/concat(.,
                            if(not(position()=last()))
                             then ' or '
                             else ()
                            )
                let $title := $f//tei:titleStmt/tei:title/translate(., '"', '')
                let $type := $f//tei:ab[@type='lido:objectWorkType']
                let $format := $f//tei:seg[@type='format']
                let $copyright := $f//tei:ab[@type="copyright"]
                let $when := $f//tei:body//tei:ab[@type="lido:subject"]//tei:date/@when/string()
                let $notBefore := $f//tei:body//tei:ab[@type="lido:subject"]//tei:date/@notBefore/string()
                let $notAfter := $f//tei:body//tei:ab[@type="lido:subject"]//tei:date/@notAfter/string()
                let $ocm := string-join(distinct-values($f//tei:term[starts-with(@key, 'ocm')]/@key/substring-after(string(), 'ocm:')), '; ')
                let $i_keywords := ''
                let $imgNumber := count($f//tei:graphic)
                let $img1 := if ($f//tei:graphic[@xml:id='IMAGE.1']) then (concat(substring-after($pid, 'o:'), '.jpg')) else ''
            (:    es gibt nur IMAGE.1! :)
                order by $pid
                return ('&#10;"',$pid,'","',$creator,'","',$title,'","',$type,'","',$format,'","',$copyright,'","',$when,'","',$notBefore,'","',$notAfter,'","',$ocm,'","',$i_keywords,'","',$imgNumber,'","',$img1,'"')
                    
        (: The following is done for each document in the collection "vismig": :)
        case "vismig" return
            for $f in $file//tei:TEI 
                let $pid := $f//tei:idno[@type="PID"]
                let $creator := $f//tei:creation/tei:persName/tei:ref
                let $title := $f//tei:titleStmt/tei:title/translate(., '"', '')
                let $type := 'Photographic negative'
                let $format := ''
                let $copyright := $f//tei:ab[@type="copyright"]
                let $when := $f//tei:date[@type="creation"]/@when/string()
                let $notBefore := $f//tei:date[@type="creation"]//@notBefore/string()
                let $notAfter := $f//tei:date[@type="creation"]/@notAfter/string()
                let $ocm := ''
                let $i_keywords := string-join($f//tei:keywords[@resp="VB"]/tei:term/@key/string(), '; ')
                let $imgNumber := count($f//tei:graphic)
                let $img1 := $f//tei:graphic[@xml:id='IMAGE.1']/substring-after(@url/string(), '../img/')
                let $img2 := $f//tei:graphic[@xml:id='IMAGE.2']/substring-after(@url/string(), '../img/')
                let $img3 := $f//tei:graphic[@xml:id='IMAGE.3']/substring-after(@url/string(), '../img/')
                let $img4 := $f//tei:graphic[@xml:id='IMAGE.4']/substring-after(@url/string(), '../img/')
                let $img5 := $f//tei:graphic[@xml:id='IMAGE.5']/substring-after(@url/string(), '../img/')
                let $img6 := $f//tei:graphic[@xml:id='IMAGE.6']/substring-after(@url/string(), '../img/')
                let $img7 := $f//tei:graphic[@xml:id='IMAGE.7']/substring-after(@url/string(), '../img/')
                let $img8 := $f//tei:graphic[@xml:id='IMAGE.8']/substring-after(@url/string(), '../img/')
                let $img9 := $f//tei:graphic[@xml:id='IMAGE.9']/substring-after(@url/string(), '../img/')
                let $img10 := $f//tei:graphic[@xml:id='IMAGE.10']/substring-after(@url/string(), '../img/')
                let $img11 := $f//tei:graphic[@xml:id='IMAGE.11']/substring-after(@url/string(), '../img/')
                let $img12 := $f//tei:graphic[@xml:id='IMAGE.12']/substring-after(@url/string(), '../img/')
                let $img13 := $f//tei:graphic[@xml:id='IMAGE.13']/substring-after(@url/string(), '../img/')
                let $img14 := $f//tei:graphic[@xml:id='IMAGE.14']/substring-after(@url/string(), '../img/')
                let $img15 := $f//tei:graphic[@xml:id='IMAGE.15']/substring-after(@url/string(), '../img/')
                let $img16 := $f//tei:graphic[@xml:id='IMAGE.16']/substring-after(@url/string(), '../img/')
                let $img17 := $f//tei:graphic[@xml:id='IMAGE.17']/substring-after(@url/string(), '../img/')
                let $img18 := $f//tei:graphic[@xml:id='IMAGE.18']/substring-after(@url/string(), '../img/')
                order by $pid
                return ('&#10;"',$pid,'","',$creator,'","',$title,'","',$type,'","',$format,'",',$copyright,'","',$when,'","',$notBefore,'","',$notAfter,'","',$ocm,'","',$i_keywords,'","',$imgNumber,'","',$img1,'","',$img2,'","',$img3,'","',$img4,'","',$img5,'","',$img6,'","',$img7,'","',$img8,'","',$img9,'","',$img10,'","',$img11,'","',$img12,'","',$img13,'","',$img14,'","',$img15,'","',$img16,'","',$img17,'","',$img18,'"')
        
        (: The following is done for each document in the collection "polos": :)
        case "polos" return
            for $f in $file//lido:lido 
                let $pid := $f//lido:recordID[@lido:type="PID"]
                let $creator := string-join($f//lido:eventSet[.//lido:term = 'Production']//lido:displayActorInRole, '; ')
                let $title := $f//lido:objectIdentificationWrap/lido:titleWrap/lido:titleSet/lido:appellationValue/translate(., '"', '')
                let $type := 'Postcard'
                let $format := $f//lido:measurementValue[@lido:label="format"]
                let $copyright := 'Creative Commons BY-NC-SA 4.0'
                let $when := ''
                let $notBefore := $f//lido:earliestDate[@lido:type="use"]
                let $notAfter := $f//lido:latestDate[@lido:type="use"]
                let $ocm := string-join(distinct-values($f//lido:conceptID[@lido:source="ocm"]), '; ')
                let $i_keywords := ''
                let $imgNumber := count($f//lido:linkResource[@lido:formatResource="image/jpeg"])
                let $img1 := $f//lido:resourceSet[@lido:sortorder="1"]//lido:linkResource[@lido:formatResource="image/jpeg"]/translate(., 'S', 's')
                let $img2 := $f//lido:resourceSet[@lido:sortorder="2"]//lido:linkResource[@lido:formatResource="image/jpeg"]/translate(., 'S', 's')
(:                translate to fix a small error in the data:)
                order by $pid
                return ('&#10;"',$pid,'","',$creator,'","',$title,'","',$type,'","',$format,'","',$copyright,'","',$when,'","',$notBefore,'","',$notAfter,'","',$ocm,'","',$i_keywords,'","',$imgNumber,'","',$img1,'","',$img2,'"')

        
        default return ''

(: Return CSV header and rows :)
return ($csv-row-headerPerFile, $csv-row)


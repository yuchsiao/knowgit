var network;

$(function(){

    var flag_graph_toggle = false;

    function sendInfo() {
        $.getJSON($SCRIPT_ROOT + '/_query', {
            repo: $('#repo-input').val()
        }, drawNetwork);
        return false;
    }


    function drawNetwork(d) {

        // create people.
        // value corresponds with the age of the person

        var nodes = d['nodes'];
        var edges = d['edges'];
        var focus_id = d['focus_id'];

        // Instantiate our network object.
        var container = document.getElementById('network-container');
        var data = {
            nodes: nodes,
            edges: edges
        };

        var options = {
            stabilize: false,
            nodes: {
                shape: 'dot',
                color: {
                    highlight: {
                        background: '#c12c2c',
                        border: '#ffd5d8'
                    },
                    background: "#1471ba",
                    border: '#CFE2FF'
                },
                fontSize: 20
            },
            edges: {
                color: '#97C2FC'
            }
        };
        network = new vis.Network(container, data, options);
        network.on('select', function (properties) {
            if (properties.nodes.length==0) {
                return;
            }
            clickOnNode(properties.nodes[0]);
        });
        network.on('doubleClick', function (properties) {
            var fullName = $('#info-container .user').text() + '/' + $('#info-container .repo').text();
            $('#repo-input').val(fullName);
            sendInfo();
        });

        network.focusOnNode(focus_id);
        network.setSelection([focus_id]);
        clickOnNode(focus_id);
        network.moveTo({position: {x:0, y:0}});
    }


    function clickOnNode(id) {
        $.getJSON($SCRIPT_ROOT + '/_repo_info', {
            repo_id: id
        }, showRepoInfo);

        function showRepoInfo(data) {
            if ($('#info-container').length > 0) {
                $('#info-container').remove();
            }
            var container = $('#main-container');

            text =
                '<div id="info-container">' +
                '<h3><span class="user"><a target="_blank" href="https://github.com/' + data['user'] + '">' +
                data['user'] + '</a></span> / ' +
                '<span class="repo"><a target="_blank" href="https://github.com/' + data['user'] + '/' + data['repo'] + '">' +
                data['repo'] + '</a></span></h3>' +
                '<p class="description">' + data['description'] + '</p>' +
                '<div id="score"><span class="repo-stat"><span class="octicon octicon-star"></span> Star </span>' +
                '<span class="repo-stat-value">' + data['star'] + '</span>  ' +
                '<span class="repo-stat"><span class="octicon octicon-repo-forked"></span> Fork </span>' +
                '<span class="repo-stat-value">' + data['fork'] + '</span></div>' +
                '</div>';

            text2 = '<div class="repo-time-container">' +
                '<span class="repo-time-title">created at ' + '</span><span class="repo-time-value">' + data['created'] + '</span>, ' +
                '<span class="repo-time-title">pushed at ' + '</span><span class="repo-time-value">' + data['pushed'] + '</span>' +
                '</div>';

            container.append(text);
            d3.select("#info-container").append("div")
                .attr("id", "readme")
                .html(data['readme']);

            var d3Container = d3.select("#info-container");
            d3Container.selectAll()
                .data(data['tags'])
                .enter()
                .append("span")
                .attr("class", "tag")
                .text(function (d) {
                    return d['text'];
                });

//            tmp = $('#wordcloud');
//            if (tmp.length>0) {
//                tmp.remove();
//            }
//            $('#info-container').append('<div id="wordcloud"></div>');
//
//            $('#wordcloud').jQCloud(data['tags']);

            $('#info-container').append(text2);

        }
    }

    $('#repo-input').bind('keyup', function(e) {
        if (e.keyCode == 13) {
            sendInfo();

            d3.select('#title').transition().duration(1000)
                .style('height', '0px')
                .style('padding', '0 0 0 0')
                .style('margin', '0 0 0 0')
                .each('end', function () {
                    d3.select('#title')
                        .style('display', 'none');
                    d3.select('#repo-input-knowgit')
                        .transition()
                        .duration(1000)
                        .style('margin-left', '-242px')
                        .style('opacity', 1);
                    d3.select('#repo-input-github')
                        .transition()
                        .duration(1000)
                        .style('margin-left', '138px')
                        .style('opacity', 1);
                });
            d3.select('#repo-input-container')
                .transition()
                .duration(1000)
                .style('top', '75px');

        }
    });

    $('#repo-input').focus();

});
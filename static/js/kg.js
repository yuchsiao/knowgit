var network;

var historyGraph = {
    nodes: [],
    edges: []
};

//var historyLength = 5;

var currentGraph;

$(function(){

    var flag_graph_toggle = false;

    function sendInfo() {
        $.getJSON($SCRIPT_ROOT + '/_query', {
            repo: $('#repo-input').val()
        }, drawNetwork);
        return false;
    }

    function drawNetwork(d) {

        currentGraph = d;

        var nodes = d['nodes'];
        var edges = d['edges'];

        // Remove duplicate edges and nodes
        for (var j=0; j<historyGraph.nodes.length-1; ++j) {
            for (var i=0; i<nodes.length; ++i) {
                historyGraph.nodes[j].id;
                if (nodes[i].id == historyGraph.nodes[j].id) {
//                    for (var k=0; k<edges.length; ) {
//                        if (edges[k].to == nodes[i].id || edges[k].from == nodes[i].id) {
//                            edges.splice(k,1);
//                            continue;
//                        }
//                        ++k;
//                    }
                    nodes.splice(i,1);
                    break;
                }
            }
        }
//        for (var j=0; j<historyGraph.nodes.length-1; ++j) {
//            for (var i=0; i<edges.length; ) {
//                if (edges[i].to == historyGraph.nodes[j].id) {
//                    edges.splice(i,1);
//                    continue;
//                }
//                ++i;
//            }
//        }
        // Remove floating nodes
//        removeFloatingNodeLoop:
//        for (var i=0; i<nodes.length; ) {
//            for (var j=0; j<edges.length; ++j) {
//                if (nodes[i].id == edges[j].to || nodes[i].id == edges[j].from) {
//                    ++i;
//                    continue removeFloatingNodeLoop;
//                }
//            }
//            nodes.splice(i,1);
//        }


        edges = edges.concat(historyGraph.edges);
        // Remove duplicate edges
        for (var i=0; i<edges.length-1; ++i) {
            for (var j=i+1; j<edges.length; ) {
                if ((edges[i].to == edges[j].to && edges[i].from == edges[j].from) ||
                    (edges[i].from == edges[j].to && edges[i].to == edges[j].from)) {
                    edges.splice(j,1);
                }
                else {
                    ++j;
                }
            }
        }


        nodesLength = nodes.length;
        nodes = nodes.concat(historyGraph.nodes.slice(0, historyGraph.nodes.length-1));
        for (var i=nodesLength; i<nodes.length; ++i) {
            nodes[i].group = 'history';
        }

        var focus_id = d['focus_id'];
        // Instantiate our network object.
        var container = document.getElementById('network-container');
        var data = {
            nodes: nodes,
            edges: edges
        };

        console.log(data);


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
            },
            groups: {
                history: {
                    shape: 'dot',
                    color: {
                        background: "#75C561",
                        border: '#D2EAD0'
                    },
                    fontSize: 20
                }
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
            if (properties.nodes.length==0) {
                return;
            }
            var fullName = $('#info-container .user').text() + '/' + $('#info-container .repo').text();
            $('#repo-input').val(fullName);
            doubleClickOnNode(focus_id, properties.nodes[0])
            sendInfo()
            console.log(historyGraph);
        });

        network.focusOnNode(focus_id);
        network.selectNodes([focus_id]);
        clickOnNode(focus_id);
        network.moveTo({position: {x:50, y:50}});
    }

    function findNode(nodes, id) {
        for (var i=0; i<nodes.length; ++i) {
            if (nodes[i].id == id) {
                return nodes[i];
            }
        }
    }

    function doubleClickOnNode(prevId, currentId) {
        if (prevId == currentId) {  // click on the same node
            return;
        }
        if (historyGraph.nodes.length==0) {
            historyGraph.nodes.push(findNode(currentGraph.nodes, prevId));
        }

        // Check if the currentId is in history already
        for (var ind=0; ind<historyGraph.edges.length; ++ind) {
            if (historyGraph.edges[ind].from == currentId) {
                historyGraph.edges = historyGraph.edges.slice(0, ind);
                historyGraph.nodes = historyGraph.nodes.slice(0, ind+1);
                return;
            }
        }

        // Prepare one-step forward and one-step backward edge lists
        var fromList = [];
        var toList = [];
        for (var i=0; i<currentGraph.edges.length; ++i) {
            if (currentGraph.edges[i].from == prevId) {
                fromList.push(currentGraph.edges[i]);
            }
            if (currentGraph.edges[i].to == currentId) {
                toList.push(currentGraph.edges[i]);
            }
        }

        // One-step case
        var flagOneStep = false;
        for (var i=0; i<fromList.length; ++i) {
            if (fromList[i].to == currentId) {
                historyGraph.edges.push(fromList[i]);
                historyGraph.nodes.push(findNode(currentGraph.nodes, currentId));
                flagOneStep = true;
                break;
            }
        }

        // Two-step case
        if (flagOneStep==false) {
            twoStepLoop:
            for (var i=0; i<fromList.length; ++i) {
                for (var j=0; j<toList.length; ++j) {
                    if (fromList[i].to == toList[j].from) {
                        historyGraph.edges.push(fromList[i]);
                        historyGraph.edges.push(toList[j]);
                        historyGraph.nodes.push(findNode(currentGraph.nodes, fromList[i].to));
                        historyGraph.nodes.push(findNode(currentGraph.nodes, toList[j].to));
                        break twoStepLoop;
                    }
                }
            }
        }

        if (flagOneStep) {
            historyGraph.edges = [historyGraph.edges[historyGraph.edges.length-1]];
            historyGraph.nodes = historyGraph.nodes.slice(historyGraph.nodes.length-2, historyGraph.nodes.length);
        }
        else {
            historyGraph.edges = historyGraph.edges.slice(historyGraph.edges.length-2, historyGraph.edges.length);
            historyGraph.nodes = historyGraph.nodes.slice(historyGraph.nodes.length-3, historyGraph.nodes.length);
        }
        console.log(historyGraph)
//        historyGraph.edges = historyGraph.edges.slice(Math.max(0, historyGraph.edges.length-2), historyGraph.edges.length);
//        historyGraph.nodes = historyGraph.nodes.slice(Math.max(0, historyGraph.nodes.length-3), historyGraph.nodes.length);

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

            $('#info-container').append(text2);

        }
    }

    $('#repo-input').bind('keyup', function(e) {
        if (e.keyCode == 13) {

            // Refresh history
            historyGraph = {
                nodes: [],
                edges: []
            };


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
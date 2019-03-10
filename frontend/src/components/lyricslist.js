import React from 'react';
import {Link} from 'react-router-dom';
import {ListGroup} from 'react-bootstrap';
import axios from "axios";

export default class LyricsList extends React.Component {

    state = {
        lyrics: []
    };

    componentDidMount() {
        this.getLyrics();
    }

    getLyrics = () => {
        axios.get("/api/lyrics/").then(result => this.setState({lyrics: result.data}))
    };

    renderLyrics() {
        const lyrics = Object.values((this.state.lyrics));
        console.log(lyrics);
        return lyrics.map(lyric => <ListGroup.Item variant="light" align="center"><Link
            to={`/lyrics/${lyric.id}`}>{lyric.title}</Link></ListGroup.Item>);
    }

    render() {
        return (
            <div>
                <ListGroup>
                    {this.renderLyrics()}
                </ListGroup>
            </div>
        )
    }

}
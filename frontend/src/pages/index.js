import React from 'react';

import LyricList from '../components/lyricslist'

export default class IndexPage extends React.Component {
    render() {
        const lyrics = Object.values(this.props.lyrics);

        return (
            <div>
                <h1>Lyrics</h1>
                <LyricList lyrics={this.props.lyrics}/>
            </div>
        )
    }

}